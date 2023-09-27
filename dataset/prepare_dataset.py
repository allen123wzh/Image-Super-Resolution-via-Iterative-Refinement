import argparse
from multiprocessing import Lock, Process, RawValue
from functools import partial
from multiprocessing.sharedctypes import RawValue
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as trans_fn
import os
from pathlib import Path
import numpy as np
import time
import cv2


def resize_and_convert(img, size, resample, center_crop=False):
    if(img.size[0] != size):
        img = trans_fn.resize(img, size, resample)
        if center_crop:
            img = trans_fn.center_crop(img, size)
    return img    


def creat_gamma_lut(gamma):
    lut = np.array([((i/255.0)**gamma)*255 for i in range(256)])
    lut = np.clip(lut, 0, 255).astype((np.uint8))
    return lut


def image_degradation(image, gamma_range):
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)  
    image_blue_degradation = image.copy()
    
    # blue channel degradation, factor range from 0.6 to 0.8
    min_factor = 0.6
    max_factor = 0.8
    blue_degradatoin_factor = np.random.uniform(min_factor,max_factor)
    # print(blue_degradatoin_factor)
    
    # blue channel degradation
    image_blue_degradation[0,:,:] = np.clip(image_blue_degradation[0,:,:]*blue_degradatoin_factor,0,255).astype(np.uint8)

    # contrast and brightness adjustment
    # alpha=1.0 means no change
    # beta=0 means no change
    contrast = np.random.uniform(0.2,0.5)
    bright = np.random.uniform(-5,5)
    contrast_brightness_adjust_image = cv2.convertScaleAbs(image_blue_degradation, alpha=contrast,beta=bright)

    # generate Gaussian noise to simulate the optical to electricity noise
    mean = 0
    std_dev = np.random.uniform(10,20)
    noise = np.random.normal(mean, std_dev, image.shape)
    noisy_image = np.clip(contrast_brightness_adjust_image+noise, 0,255).astype(np.uint8)

    # gamma degradation of the image
    gamma=np.random.uniform(gamma_range[0],gamma_range[1])
    lut = creat_gamma_lut(gamma)
    gamma_image = cv2.LUT(noisy_image,lut)

    # return Image.fromarray(cv2.cvtColor(gamma_image,cv2.COLOR_BGR2RGB))


    #jpeg degradation of the image, range is 50~100
    jpeg_quality = np.random.randint(85,95)
    jpeg_compression_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    degradation_image_name = './temp.jpeg'
    cv2.imwrite(degradation_image_name, gamma_image, params=jpeg_compression_params)
    compressed_image = cv2.imread(degradation_image_name)

    return Image.fromarray(cv2.cvtColor(compressed_image,cv2.COLOR_BGR2RGB))


    # cv2.imshow("blue degradation", image_blue_degradation)
    # cv2.imshow("contrast_degradation", contrast_brightness_adjust_image)
    # cv2.imshow("Gaussian_degradatoin", noisy_image)
    # cv2.imshow("gamma_degradation", gamma_image)
    # cv2.imshow("jpeg", compressed_image)
    # cv2.imshow("original", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def resize_multiple(img, sizes=(16, 128), resample=Image.BICUBIC,
                    center_crop=False, 
                    degradation=False,
                    gamma_range=(1.5,1.7)):
    # Low res degraded image
    lr_img = resize_and_convert(img, sizes[0], resample, center_crop)
    if degradation:
        lr_img = image_degradation(lr_img, gamma_range)
    # Original high res image
    hr_img = resize_and_convert(img, sizes[1], resample, center_crop)
    
    # Upsampled low-res degraded image for NN input
    sr_img = resize_and_convert(lr_img, sizes[1], resample, center_crop)
    
    # Histogram equalized low-light image
    hiseq_img = trans_fn.equalize(sr_img)

    return [lr_img, hr_img, sr_img, hiseq_img]


def resize_worker(img_file, sizes, resample,
                  center_crop = False,
                  degradation = False,
                  gamma_range = (1.5, 1.7)):

    img = Image.open(img_file)
    img = img.convert('RGB')
    out = resize_multiple(
        img, sizes=sizes, resample=resample, 
        center_crop = center_crop,
        degradation = degradation,
        gamma_range=gamma_range)
    
    filename = img_file.name.split('.')[0]
    filename = filename[-8:-5]

    return filename, out


class WorkingContext():
    def __init__(self, resize_fn, out_path, sizes):
        self.resize_fn = resize_fn
        # self.lmdb_save = lmdb_save
        self.out_path = out_path
        # self.env = env
        self.sizes = sizes

        self.counter = RawValue('i', 0)
        self.counter_lock = Lock()

    def inc_get(self):
        with self.counter_lock:
            self.counter.value += 1
            return self.counter.value

    def value(self):
        with self.counter_lock:
            return self.counter.value


def prepare_process_worker(wctx, file_subset):
    for file in file_subset:
        i, imgs = wctx.resize_fn(file)
        lr_img, hr_img, sr_img, hiseq_img = imgs

        lr_img.save(
            '{}/lr/{}.png'.format(wctx.out_path, i.zfill(5)))
        hr_img.save(
            '{}/hr/{}.png'.format(wctx.out_path, i.zfill(5)))
        sr_img.save(
            '{}/sr/{}.png'.format(wctx.out_path, i.zfill(5)))
        hiseq_img.save(
            '{}/hiseq/{}.png'.format(wctx.out_path, i.zfill(5)))
        
        # lr_img.save(
        #     '{}/lr/{}.png'.format(wctx.out_path, str(wctx.value()).zfill(5)))
        # hr_img.save(
        #     '{}/hr/{}.png'.format(wctx.out_path, str(wctx.value()).zfill(5)))
        # sr_img.save(
        #     '{}/sr/{}.png'.format(wctx.out_path, str(wctx.value()).zfill(5)))
        # hiseq_img.save(
        #     '{}/hiseq/{}.png'.format(wctx.out_path, str(wctx.value()).zfill(5)))

        wctx.inc_get()


def all_threads_inactive(worker_threads):
    for thread in worker_threads:
        if thread.is_alive():
            return False
    return True


def prepare(img_path, out_path, n_worker, sizes=(16, 128), resample=Image.BICUBIC, 
            center_crop=False, degradation=False, gamma_range=(1.5, 1.7)):
    
    resize_fn = partial(resize_worker, sizes=sizes,
                        resample=resample,
                        center_crop = center_crop,
                        degradation = degradation,
                        gamma_range=gamma_range)
    
    files = [p for p in Path(
        '{}'.format(img_path)).glob(f'**/*')]

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(f'{out_path}/lr', exist_ok=True)
    os.makedirs(f'{out_path}/hr', exist_ok=True)
    os.makedirs(f'{out_path}/sr', exist_ok=True)
    os.makedirs(f'{out_path}/hiseq', exist_ok=True)

    if n_worker > 1:


        file_subsets = np.array_split(files, n_worker)
        worker_threads = []
        wctx = WorkingContext(resize_fn, out_path, sizes)

        # start worker processes, monitor results
        for i in range(n_worker):
            proc = Process(target=prepare_process_worker, args=(wctx, file_subsets[i]))
            proc.start()
            worker_threads.append(proc)
        
        total_count = str(len(files))
        while not all_threads_inactive(worker_threads):
            print("\r{}/{} images processed".format(wctx.value(), total_count), end=" ")
            time.sleep(0.1)

    else:
        total = 0
        for file in tqdm(files):
            i, imgs = resize_fn(file)
            lr_img, hr_img, sr_img, hiseq_img = imgs
            
            lr_img.save(
                '{}/lr/{}.png'.format(out_path, i.zfill(5)))
            hr_img.save(
                '{}/hr/{}.png'.format(out_path, i.zfill(5)))
            sr_img.save(
                '{}/sr/{}.png'.format(out_path, i.zfill(5)))
            hiseq_img.save(
                '{}/hiseq/{}.png'.format(out_path, i.zfill(5)))
            
            # lr_img.save(
            #     '{}/lr/{}.png'.format(out_path, str(total).zfill(5)))
            # hr_img.save(
            #     '{}/hr/{}.png'.format(out_path, str(total).zfill(5)))
            # sr_img.save(
            #     '{}/sr/{}.png'.format(out_path, str(total).zfill(5)))
            # hiseq_img.save(
            #     '{}/hiseq/{}.png'.format(out_path, str(total).zfill(5)))

            total += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str,
                        default='/home/allen/Documents/MIE288/sr3_server4/data/dark/jenny/test_512/rgb_512'.format(Path.home()))
    parser.add_argument('--out', '-o', type=str,
                        default='./data/dark/jenny_test')
    parser.add_argument('--size', type=str, default='256,256')  # shorter edge
    parser.add_argument('--n_worker', type=int, default=1)
    parser.add_argument('--resample', type=str, default='bicubic')
    
    parser.add_argument('--center_crop', '-c', action='store_true')
    parser.add_argument('--degradation', '-d', action='store_true')
    parser.add_argument('--gamma', '-g', type=str, default='1.3,1.5') # low-light gamma division (larger number, darker img)

    args = parser.parse_args()

    resample_map = {'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
    resample = resample_map[args.resample]
    sizes = [int(s.strip()) for s in args.size.split(',')]
    
    if not args.degradation:
        args.out = '{}_{}_{}'.format(args.out, sizes[0], sizes[1])
        
        prepare(args.path, args.out, args.n_worker,
            sizes=sizes, resample=resample, 
            center_crop=args.center_crop,
            degradation=False)
    else:
        gamma_range = [float(s.strip()) for s in args.gamma.split(',')]

        args.out = '{}_{}_{}_gamma_{}_{}'.format(args.out, sizes[0], sizes[1], 
                                                gamma_range[0], gamma_range[1])
        prepare(args.path, args.out, args.n_worker,
                sizes=sizes, resample=resample, 
                center_crop=args.center_crop,
                degradation=args.degradation,
                gamma_range=gamma_range)
