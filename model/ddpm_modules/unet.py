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


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, dropout=0, norm_groups=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            # Swish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        if exists(self.mlp):
            h += self.mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class MobileNetV3Block(nn.Module):
    def __init__(self, in_channels, out_channels, norm_groups):
        super().__init__()
        # self.residual = False
        self.residual = in_channels==out_channels
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=int(in_channels)),
            # nn.GroupNorm(norm_groups, in_channels),
            nn.SiLU(),
        )
        self.se_block = SqueezeExcitation(input_channels=in_channels, 
                                          squeeze_channels=in_channels//8,
                                          activation=nn.SiLU,
                                          scale_activation=nn.Sigmoid)
        self.pw_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            # nn.GroupNorm(norm_groups, out_channels)
        )

    def forward(self, x):
        h = self.pw_conv(self.se_block(self.dw_conv(x)))
        # h = self.pw_conv(self.dw_conv(x))
        # return h
        if self.residual:
            return h+x
        else:
            return h


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, feat_dim, emb_dim, norm_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(norm_groups, feat_dim)
        self.emb = nn.Linear(emb_dim, feat_dim*2)
    
    def forward(self, feat, time_emb):
        emb = self.emb(time_emb)[:,:,None,None]
        ys, yb = torch.chunk(emb, 2, dim=1)
        return (1+ys) * self.norm(feat) + yb



class BigGANResBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, dropout=0, norm_groups=32,
                 up=False, down=False, dwconv=False, bneck=True, AdaGN_all=True):
        super().__init__()
        
        self.AdaGN_all = AdaGN_all
        self.norm_groups = norm_groups
        self.up_block = Upsample() if up else None
        self.down_block = Downsample() if down else None

        if not dwconv:
            ### BigGAN structure ###
            if not bneck:
                # 1. Norm
                self.in_norm = nn.Sequential(
                    nn.GroupNorm(norm_groups, dim),
                    nn.SiLU(),
                )
                # 2. Up/Down/No-sample
                # 3. First 3x3 Conv
                self.in_conv = nn.Conv2d(dim, dim_out, 3, padding=1)
                # 4. Norm
                self.out_norm = nn.GroupNorm(norm_groups, dim_out)
                # 5. AdaGN, inject timestep embedding to norm(h)
                self.emb_layers = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_emb_dim, 2 * dim_out)
                )
                # 6. Second 3x3 Conv 
                self.out_conv = nn.Sequential(
                    nn.SiLU(),
                    nn.Dropout(p=dropout),
                    nn.Conv2d(dim_out, dim_out, 3, padding=1), 
                ) 

            ### BigGAN-deep structure ###
            else:
                if not AdaGN_all:
                    hidden_dim = dim//4
                    hidden_norm_groups = hidden_dim//4
                    # 1. Norm and bneck channel downsample
                    self.in_norm = nn.Sequential(
                        nn.GroupNorm(norm_groups, dim),
                        nn.SiLU(),
                        nn.Conv2d(dim, hidden_dim, 1),    # 1x1 bneck down
                        nn.GroupNorm(hidden_norm_groups, hidden_dim),
                        nn.SiLU(),
                    )   
                    # 2. Up/Down/No-sample
                    # 3. First 3x3 Conv
                    self.in_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
                    # 4. Norm
                    self.out_norm = nn.GroupNorm(hidden_norm_groups, hidden_dim)
                    # 5. AdaGN, inject timestep embedding to norm(h)
                    self.emb_layers = nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(time_emb_dim, 2 * hidden_dim)
                    )
                    # 6. Second 3x3 Conv and bneck channel upsample
                    self.out_conv = nn.Sequential(
                        nn.SiLU(),
                        nn.Dropout(p=dropout),
                        nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), 
                        nn.GroupNorm(hidden_norm_groups, hidden_dim),
                        nn.SiLU(),
                        nn.Conv2d(hidden_dim, dim_out, 1)   # 1x1 bneck up
                    ) 
                else:
                    hidden_dim = dim//4
                    hidden_norm_groups = norm_groups//4
                    # 1. AdaGN + Act + 1x1 Conv down
                    self.norm1 = AdaptiveGroupNorm(dim, time_emb_dim, norm_groups)
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

        else:
            self.in_norm = nn.Sequential(
                nn.GroupNorm(norm_groups, dim),
                nn.SiLU(),
            )
            self.in_conv = MobileNetV3Block(dim, dim_out, norm_groups)

            self.out_norm = nn.Sequential(
                nn.GroupNorm(norm_groups, dim_out),
            )
            self.out_conv = nn.Sequential(
                nn.SiLU(),
                nn.Dropout(p=dropout),
                MobileNetV3Block(dim_out, dim_out, norm_groups)
            )

        if dim == dim_out:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(dim, dim_out, 1)
    
    def forward(self, x, time_emb): # x: BCHW, time: BC
        if not self.AdaGN_all:
            # BigGAN first 3x3 conv block
            if self.up_block:
                h = self.in_norm(x)
                h = self.up_block(h)
                h = self.in_conv(h)
                x = self.up_block(x)
            elif self.down_block:
                h = self.in_norm(x)
                h = self.down_block(h)
                h = self.in_conv(h)
                x = self.down_block(x)
            else:
                h = self.in_conv(self.in_norm(x))
            # AdaGN for time embedding
            emb = self.emb_layers(time_emb)[:, :, None, None]
            ys, yb = torch.chunk(emb, 2, dim=1)
            h = (1+ys) * self.out_norm(h) + yb
            # BigGAN second 3x3 conv block
            h = self.out_conv(h)
            # Residual add
            return self.skip_connection(x)+h
        else: 
            h = self.conv1(self.norm1(x, time_emb))
            if self.up_block:
                h = self.conv2[:1](self.norm2(h, time_emb))
                h = self.up_block(h)
                h = self.conv2[1:](h)
                x = self.up_block(x)
            elif self.down_block:
                h = self.conv2[:1](self.norm2(h, time_emb))
                h = self.down_block(h)
                h = self.conv2[1:](h)
                x = self.down_block(x)
            else:
                h = self.conv2(self.norm2(h, time_emb))
            h = self.conv3(self.norm3(h, time_emb))
            h = self.conv4(self.norm4(h, time_emb))

            return self.skip_connection(x)+h



class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape

        n_head = 4
        # n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0, 
                 with_attn=False, biggan=False, up=False, down=False):
        super().__init__()
        self.with_attn = with_attn
        
        if biggan:
            self.res_block = BigGANResBlock(
                dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout,
                up=up, down=down
            )
        else:
            self.res_block = ResnetBlock(
                dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout
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
        # image_size=128
    ):
        super().__init__()

        if with_time_emb:
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                nn.SiLU(),
                # Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            time_dim = None
            self.time_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        # now_res = image_size
        downsample_rate = 1

        ##### Downsample blocks
        downs = [nn.Conv2d(in_channel, inner_channel, 3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            
            # use_attn = (now_res in attn_res)
            use_attn = (downsample_rate in attn_res)
            
            channel_mult = inner_channel * channel_mults[ind]
            
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, time_emb_dim=time_dim, 
                    norm_groups=norm_groups, dropout=dropout, with_attn=use_attn,
                    biggan=True, down=False))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            
            if not is_last:
                # downs.append(Downsample(pre_channel))
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, time_emb_dim=time_dim, 
                    norm_groups=norm_groups, dropout=dropout, with_attn=False,
                    biggan=True, down=True))                
                feat_channels.append(pre_channel)
                # now_res = now_res//2
                downsample_rate *= 2

        self.downs = nn.ModuleList(downs)

        ##### Middle blocks
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, time_emb_dim=time_dim, 
                               norm_groups=norm_groups, dropout=dropout, with_attn=True,
                               biggan=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, time_emb_dim=time_dim, 
                               norm_groups=norm_groups, dropout=dropout, with_attn=False,
                               biggan=True)
        ])
        

        ##### Upsample blocks
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)

            # use_attn = (now_res in attn_res)
            use_attn = (downsample_rate in attn_res)
            
            channel_mult = inner_channel * channel_mults[ind]

            for i in range(res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, time_emb_dim=time_dim, 
                     norm_groups=norm_groups, dropout=dropout, with_attn=use_attn,
                     biggan=True, up=False))
                pre_channel = channel_mult
            
            # if not is_last:
                if not is_last and i == res_blocks:
                    # ups.append(Upsample(pre_channel))
                    ups.append(ResnetBlocWithAttn(
                        pre_channel, channel_mult, time_emb_dim=time_dim, 
                        norm_groups=norm_groups, dropout=dropout, with_attn=False,
                        biggan=True, up=True))      
                    # now_res = now_res*2
                    downsample_rate /= 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time):
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        feats = []
        for i, layer in enumerate(self.downs):
            # print(f'down block {i}')
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                if not layer.res_block.up_block:
                    x = layer(torch.cat((x, feats.pop()), dim=1), t)
                else:
                    x = layer(x, t)
            else:
                x = layer(x)

        return self.final_conv(x)
