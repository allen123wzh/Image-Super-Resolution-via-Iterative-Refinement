from PIL import Image
import torch
from torch.utils.data import Dataset
import dataset.data_util as Util
import torchvision.transforms as T
import torchvision.transforms.functional as F
from pathlib import Path

class LRHRDataset(Dataset):
    def __init__(self, dataroot, split='train', 
                 hr_mean=[0.5, 0.5, 0.5], hr_std=[0.5, 0.5, 0.5], 
                 lr_mean=[0.0789, 0.0594, 0.0520], lr_std=[0.0754, 0.0638, 0.0614], # FFHQ_LL
                 ir_mean=[0.44], ir_std=[0.1],
                 ir = False,
                 instance_norm = True,
                 data_len=-1):

        self.data_len = data_len
        self.split = split # 'train' 'val' 'test'
        self.ir = ir
        self.instance_norm = instance_norm

        self.lr_path = Util.get_paths_from_images(f'{dataroot}/lr')

        if self.split != 'test':
            self.hr_path = Util.get_paths_from_images(f'{dataroot}/hr')    
        
        if self.ir:
            self.ir_path = Util.get_paths_from_images(f'{dataroot}/ir')

        self.lr_transform = T.Compose([T.ToTensor(),
                                       T.Normalize(lr_mean, lr_std)])
        self.hr_transform = T.Compose([T.ToTensor(),
                                       T.Normalize(hr_mean, hr_std)])
        self.ir_transform = T.Compose([T.ToTensor(),
                                       T.Normalize(ir_mean, ir_std)])

        self.dataset_len = len(self.lr_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)
        

    def find_mean_std(self, img_tensor, per_channel=True): # img 3-channel (C,H,W)
        mean_ls = []
        std_ls = []
        channels = img_tensor.shape[0]

        if per_channel:
            for i in range(channels):
                pixels_per_channel = img_tensor[i,:,:]
                std, mean = torch.std_mean(pixels_per_channel)
                mean_ls.append(mean.item())
                std_ls.append(std.item())
        else:
            std, mean = torch.std_mean(img_tensor[:,:,:])
            mean_ls.append(mean.item())
            std_ls.append(std.item())
        
        # print(f'mean:{mean_ls}, var:{std_ls}')
        return mean_ls, std_ls


    def __len__(self):
        return self.data_len


    def __getitem__(self, index):

        img_orig_LR = Image.open(self.lr_path[index]).convert("RGB")
        fname = Path(self.lr_path[index]).stem

        ##################
        if self.instance_norm:
            img_LR = F.to_tensor(img_orig_LR)
            ins_mean, ins_std = self.find_mean_std(img_LR)
            img_LR = F.normalize(img_LR, ins_mean, ins_std)
        else:
            img_LR = self.lr_transform(img_orig_LR)
        
        img_orig_LR = self.hr_transform(img_orig_LR)

        if self.split != 'test':
            img_HR = Image.open(self.hr_path[index]).convert("RGB")  # HR img
            img_HR = self.hr_transform(img_HR)

            if self.ir:
                # img_IR = Image.open(self.ir_path[index]).convert("RGB")
                img_IR = Image.open(self.ir_path[index]).convert("L")
                if self.instance_norm:
                    img_IR = F.to_tensor(img_IR)
                    ins_mean, ins_std = self.find_mean_std(img_IR)
                    img_IR = F.normalize(img_IR, ins_mean, ins_std)
                else:
                    img_IR = self.ir_transform(img_IR)
                return {'HR': img_HR, 'LR': img_LR, 'IR': img_IR, 'Index': index, "fname": fname}
            else:
                return {'HR': img_HR, 'LR': img_LR, 'Index': index, "fname": fname}
        else:
            if self.ir:
                # img_IR = Image.open(self.ir_path[index]).convert("RGB")
                img_IR = Image.open(self.ir_path[index]).convert("L")
                if self.instance_norm:
                    img_IR = F.to_tensor(img_IR)
                    ins_mean, ins_std = self.find_mean_std(img_IR)
                    img_IR = F.normalize(img_IR, ins_mean, ins_std)
                else:
                    img_IR = self.ir_transform(img_IR)
                return {'LR': img_LR, 'IR': img_IR, 'Index': index, "fname": fname}
            else:
                return {'LR': img_LR, 'Index': index, "fname": fname}

        