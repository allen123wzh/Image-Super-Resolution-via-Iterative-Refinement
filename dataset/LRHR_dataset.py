from PIL import Image
from torch.utils.data import Dataset
import dataset.data_util as Util
import torchvision.transforms as T

class LRHRDataset(Dataset):
    def __init__(self, dataroot, split='train', data_len=-1):
        # self.datatype = datatype
        # self.l_res = l_resolution
        # self.r_res = r_resolution
        self.data_len = data_len
        # self.need_LR = need_LR
        self.split = split # 'train' 'val' 'test'

        # self.sr_path = Util.get_paths_from_images(
        #     '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
        # self.hr_path = Util.get_paths_from_images(f'{dataroot}/hr')
        # if self.need_LR:
        #     self.lr_path = Util.get_paths_from_images(
        #         '{}/lr_{}'.format(dataroot, l_resolution))

        self.lr_path = Util.get_paths_from_images(f'{dataroot}/lr')

        if self.split != 'test':
            self.hr_path = Util.get_paths_from_images(f'{dataroot}/hr')

        # self.hiseq_path = Util.get_paths_from_images(
        #     '{}/hiseq_{}'.format(dataroot, r_resolution))      

        self.dataset_len = len(self.lr_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)


    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # img_HR = None
        # img_LR = None

        img_LR = Image.open(self.lr_path[index]).convert("RGB")
        img_hiseq = T.functional.equalize(img_LR)

        if self.split != 'test':
            img_HR = Image.open(self.hr_path[index]).convert("RGB")  # HR img
            
            [img_HR, img_LR, img_hiseq] = Util.transform_augment(
                [img_HR, img_LR, img_hiseq], split=self.split, min_max=(-1, 1))
            
            return {'HR': img_HR, 'SR': img_LR, 'LR': img_LR, 'hiseq': img_hiseq, 'Index': index}
        else:
            [img_LR, img_hiseq] = Util.transform_augment(
                [img_LR, img_hiseq], split=self.split, min_max=(-1, 1))
            
            return {'SR': img_LR, 'LR': img_LR, 'hiseq': img_hiseq, 'Index': index}            


        # img_SR = img_LR
        
        # img_SR = Image.open(self.sr_path[index]).convert("RGB")  # LR upsample to HR img
        #                                                             # apply hiseq to SR
        
        

        # if self.need_LR:
        #     img_LR = Image.open(self.lr_path[index]).convert("RGB")

        # if self.need_LR:
        #     [img_LR, img_SR, img_HR, img_hiseq] = Util.transform_augment(
        #         [img_LR, img_SR, img_HR, img_hiseq], split=self.split, min_max=(-1, 1))
        #     # return {'LR': img_LR, 'HR': img_HR, 'SR': img_hiseq, 'Index': index}
        #     return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'hiseq': img_hiseq, 'Index': index}
        # else:
        #     [img_SR, img_HR, img_hiseq] = Util.transform_augment(
        #         [img_SR, img_HR, img_hiseq], split=self.split, min_max=(-1, 1))
        #     # return {'HR': img_HR, 'SR': img_hiseq, 'Index': index}
        #     return {'HR': img_HR, 'SR': img_SR, 'hiseq': img_hiseq, 'Index': index}
        
