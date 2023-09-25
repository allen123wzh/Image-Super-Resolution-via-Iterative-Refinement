from PIL import Image
from torch.utils.data import Dataset
import dataset.data_util as Util
import torchvision.transforms as T
import torchvision.transforms.functional as F

class LRHRDataset(Dataset):
    def __init__(self, dataroot, split='train', 
                 hr_mean=[0.5, 0.5, 0.5], hr_std=[0.5, 0.5, 0.5], 
                 lr_mean=[0.0789, 0.0594, 0.0520], lr_std=[0.0754, 0.0638, 0.0614], # FFHQ_LL
                 data_len=-1):

        self.data_len = data_len
        self.split = split # 'train' 'val' 'test'

        self.lr_path = Util.get_paths_from_images(f'{dataroot}/lr')

        self.lr_transform = T.Compose([T.ToTensor(),
                                       T.Normalize(lr_mean, lr_std)])
        self.hr_transform = T.Compose([T.ToTensor(),
                                       T.Normalize(hr_mean, hr_std)])

        if self.split != 'test':
            self.hr_path = Util.get_paths_from_images(f'{dataroot}/hr')    

        self.dataset_len = len(self.lr_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)
        

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        img_orig_LR = Image.open(self.lr_path[index]).convert("RGB")
        
        img_LR = self.lr_transform(img_orig_LR)
        img_orig_LR = self.hr_transform(img_orig_LR)

        # img_hiseq = T.functional.equalize(img_LR)

        if self.split != 'test':
            img_HR = Image.open(self.hr_path[index]).convert("RGB")  # HR img
            img_HR = self.hr_transform(img_HR)

            return {'HR': img_HR, 'SR': img_LR, 'LR': img_LR, 'Index': index}

            # [img_HR, img_LR, img_hiseq] = Util.transform_augment(
            #     [img_HR, img_LR, img_hiseq], split=self.split, min_max=(-1, 1))
            
            # return {'HR': img_HR, 'SR': img_LR, 'LR': img_LR, 'hiseq': img_hiseq, 'Index': index}
        else:
            return {'SR': img_LR, 'LR': img_orig_LR, 'Index': index}
            
            # [img_LR, img_hiseq] = Util.transform_augment(
            #     [img_LR, img_hiseq], split=self.split, min_max=(-1, 1))
            
            # return {'SR': img_LR, 'LR': img_LR, 'hiseq': img_hiseq, 'Index': index}            

        
