import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import SqueezeExcitation
from inspect import isfunction

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# model
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.up(x)
        # return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.down(x)
        # return self.conv(x)


# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            nn.SiLU(),
            # Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, feat_dim, emb_dim, norm_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(norm_groups, feat_dim)
        self.emb = nn.Sequential(nn.SiLU(),
                                 nn.Linear(emb_dim, feat_dim*2))
    
    def forward(self, feat, time_emb):
        emb = self.emb(time_emb)[:,:,None,None]
        ys, yb = torch.chunk(emb, 2, dim=1)
        return (1+ys) * self.norm(feat) + yb



class ResBlock(nn.Module):
    # def __init__(self, dim, dim_out, time_emb_dim, dropout=0, norm_groups=32,
    #              up=False, down=False, dwconv=False, bneck=True, AdaGN_all=True, AdaGN_3=True):
    def __init__(self, dim, dim_out, time_emb_dim, dropout=0, norm_groups=32,
                 up=False, down=False, bneck_ratio=4):

        super().__init__()
        
        self.norm_groups = norm_groups
        self.up_block = Upsample() if up else None
        self.down_block = Downsample() if down else None

        hidden_dim = dim//bneck_ratio
        hidden_norm_groups = norm_groups//bneck_ratio
        # 1. GN + Act + 1x1 Conv down
        self.norm1 = nn.GroupNorm(norm_groups, dim)
        self.conv1 = nn.Sequential(nn.SiLU(),
                                nn.Conv2d(dim, hidden_dim, 1, padding=0))
        # 2. AdaGN + Act + Up/Downsample + 3x3 Conv
        self.norm2 = AdaptiveGroupNorm(hidden_dim, time_emb_dim, hidden_norm_groups)
        self.conv2 = nn.Sequential(nn.SiLU(), 
                                nn.Dropout(p=dropout),
                                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1))
        # 3. AdaGN + Act + 3x3 Conv
        self.norm3 = AdaptiveGroupNorm(hidden_dim, time_emb_dim, hidden_norm_groups)
        self.conv3 = nn.Sequential(nn.SiLU(),
                                nn.Dropout(p=dropout),
                                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1))
        # 4. AdaGN + Act + 1x1 Conv up
        self.norm4 = AdaptiveGroupNorm(hidden_dim, time_emb_dim, hidden_norm_groups)
        self.conv4 = nn.Sequential(nn.SiLU(),
                                nn.Conv2d(hidden_dim, dim_out, 1, padding=0))
        if dim == dim_out:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(dim, dim_out, 1)
    
    def forward(self, x, time_emb): # x: BCHW, time: BC
        h = self.conv1(self.norm1(x))
        if self.up_block:
            h = self.conv2[:1](self.norm2(h, time_emb))
            # h = self.up_block(h)
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = self.conv2[1:](h)
            # x = self.up_block(x)
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        elif self.down_block:
            h = self.conv2[:1](self.norm2(h, time_emb))
            # h = self.down_block(h)
            h = F.avg_pool2d(h, kernel_size=2, stride=2)
            h = self.conv2[1:](h)
            # x = self.down_block(x)
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        else:
            h = self.conv2(self.norm2(h, time_emb))
        h = self.conv3(self.norm3(h, time_emb))
        h = self.conv4(self.norm4(h, time_emb))

        return self.skip_connection(x)+h



class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=4, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape

        head_dim = channel // self.n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, self.n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, self.n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, self.n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResBlockWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0, 
                 with_attn=False, up=False, down=False):
        super().__init__()
        self.with_attn = with_attn
        
        self.res_block = ResBlock(
            dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout,
            up=up, down=down
        )
        
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(16),
        res_blocks=3,
        dropout=0,
        with_time_emb=True,
    ):
        super().__init__()

        if with_time_emb:
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                nn.SiLU(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            time_dim = None
            self.time_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        downsample_rate = 1

        ##### Downsample blocks
        downs = [nn.Conv2d(in_channel, inner_channel, 3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            
            use_attn = (downsample_rate in attn_res)
            
            channel_mult = inner_channel * channel_mults[ind]
            
            for _ in range(0, res_blocks):
                downs.append(ResBlockWithAttn(
                    pre_channel, channel_mult, time_emb_dim=time_dim, 
                    norm_groups=norm_groups, dropout=dropout, with_attn=use_attn,
                    down=False))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            
            if not is_last:
                downs.append(ResBlockWithAttn(
                    pre_channel, channel_mult, time_emb_dim=time_dim, 
                    norm_groups=norm_groups, dropout=dropout, with_attn=False,
                    down=True))                
                feat_channels.append(pre_channel)
                downsample_rate *= 2

        self.downs = nn.ModuleList(downs)

        ##### Middle blocks
        self.mid = nn.ModuleList([
            ResBlockWithAttn(pre_channel, pre_channel, time_emb_dim=time_dim, 
                               norm_groups=norm_groups, dropout=dropout, with_attn=True,
                               ),
            ResBlockWithAttn(pre_channel, pre_channel, time_emb_dim=time_dim, 
                               norm_groups=norm_groups, dropout=dropout, with_attn=False,
                               )
        ])
        

        ##### Upsample blocks
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)

            use_attn = (downsample_rate in attn_res)
            
            channel_mult = inner_channel * channel_mults[ind]

            for i in range(res_blocks+1):
                ups.append(ResBlockWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, time_emb_dim=time_dim, 
                     norm_groups=norm_groups, dropout=dropout, with_attn=use_attn,
                     up=False))
                pre_channel = channel_mult
            
                if not is_last and i == res_blocks:
                    ups.append(ResBlockWithAttn(
                        pre_channel, channel_mult, time_emb_dim=time_dim, 
                        norm_groups=norm_groups, dropout=dropout, with_attn=False,
                        up=True))      
                    downsample_rate /= 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time):
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResBlockWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResBlockWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResBlockWithAttn):
                if not layer.res_block.up_block:
                    x = layer(torch.cat((x, feats.pop()), dim=1), t)
                else:
                    x = layer(x, t)
            else:
                x = layer(x)

        return self.final_conv(x)
