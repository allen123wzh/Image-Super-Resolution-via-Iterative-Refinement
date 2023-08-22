import argparse
from io import BytesIO
import multiprocessing
from multiprocessing import Lock, Process, RawValue
from functools import partial
from multiprocessing.sharedctypes import RawValue
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as trans_fn
import os
from pathlib import Path
import lmdb
import numpy as np
import time
import cv2

def resize_and_convert(img, size, resample):
    if(img.size[0] != size):
        img = trans_fn.resize(img, size, resample)
        img = trans_fn.center_crop(img, size)
    return img    

def creat_gamma_lut(gamma):
    lut = np.array([((i/255.0)**gamma)*255 for i in range(256)])
    lut = np.clip(lut, 0, 255).astype((np.uint8))
    return lut

def image_degradation(image, gamma_range):
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)  
    image_blue_degradation = image.copy()
    #blue channel degradation, factor range from 0.6 to 0.8
    min_factor = 0.6
    max_factor = 0.8
    blue_degradatoin_factor = np.random.uniform(min_factor,max_factor)
    # print(blue_degradatoin_factor)
    #blue channel degradation
    image_blue_degradation[0,:,:] = np.clip(image_blue_degradation[0,:,:]*blue_degradatoin_factor,0,255).astype(np.uint8)

    #contrast and brightness adjustment
    #alpha=1.0 means no change
    #beta=0 means no change
    contrast = np.random.uniform(0.2,0.5)
    bright = np.random.uniform(-5,5)
    contrast_brightness_adjust_image = cv2.convertScaleAbs(image_blue_degradation, alpha=contrast,beta=bright)


    #generate Gaussian noise to simulate the optical to electricity noise
    mean = 0
    std_dev = np.random.uniform(10,20)
    noise = np.random.normal(mean, std_dev, image.shape)
    noisy_image = np.clip(contrast_brightness_adjust_image+noise, 0,255).astype(np.uint8)


    #gamma degradation of the image
    gamma=np.random.uniform(gamma_range[0],gamma_range[1])
    # gamma=np.random.uniform(1.5,1.7)
    lut = creat_gamma_lut(gamma)
    gamma_image = cv2.LUT(noisy_image,lut)

    return Image.fromarray(cv2.cvtColor(gamma_image,cv2.COLOR_BGR2RGB))

    # #jpeg degradation of the image, range is 50~100
    # jpeg_quality = np.random.randint(40,95)
    # jpeg_compression_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    # cv2.imwrite(degradation_image_name, gamma_image, params=jpeg_compression_params)
    # compressed_image = cv2.imread(degradation_image_name)    
    # cv2.imshow("blue degradation", image_blue_degradation)
    # cv2.imshow("contrast_degradation", contrast_brightness_adjust_image)
    # cv2.imshow("Gaussian_degradatoin", noisy_image)
    # cv2.imshow("gamma_degradation", gamma_image)
    # cv2.imshow("jpeg", compressed_image)
    # cv2.imshow("original", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def image_convert_bytes(img):
    buffer = BytesIO()
    img.save(buffer, format='png')
    return buffer.getvalue()


def resize_multiple(img, sizes=(16, 128), resample=Image.BICUBIC, lmdb_save=False, gamma_range=(1.5,1.7)):
    # Low res degraded image
    lr_img = resize_and_convert(img, sizes[0], resample)
    # lr_img = image_degradation(lr_img, gamma_range)
    # Original high res image
    hr_img = resize_and_convert(img, sizes[1], resample)
    # Upsampled low-res degraded image for NN input
    sr_img = resize_and_convert(lr_img, sizes[1], resample)
    # Histogram equalized low-light image
    hiseq_img = trans_fn.equalize(sr_img)

    if lmdb_save:
        lr_img = image_convert_bytes(lr_img)
        hr_img = image_convert_bytes(hr_img)
        sr_img = image_convert_bytes(sr_img)
        hiseq_img = image_convert_bytes(hiseq_img)

    return [lr_img, hr_img, sr_img, hiseq_img]

def resize_worker(img_file, sizes, resample, lmdb_save=False,
                  gamma_range=(1.5, 1.7)):

    img = Image.open(img_file)
    img = img.convert('RGB')
    out = resize_multiple(
        img, sizes=sizes, resample=resample, lmdb_save=lmdb_save, gamma_range=gamma_range)

    return img_file.name.split('.')[0], out

class WorkingContext():
    def __init__(self, resize_fn, lmdb_save, out_path, env, sizes):
        self.resize_fn = resize_fn
        self.lmdb_save = lmdb_save
        self.out_path = out_path
        self.env = env
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
        if not wctx.lmdb_save:
            lr_img.save(
                '{}/lr_{}/{}.png'.format(wctx.out_path, wctx.sizes[0], i.zfill(5)))
            hr_img.save(
                '{}/hr_{}/{}.png'.format(wctx.out_path, wctx.sizes[1], i.zfill(5)))
            sr_img.save(
                '{}/sr_{}_{}/{}.png'.format(wctx.out_path, wctx.sizes[0], wctx.sizes[1], i.zfill(5)))
            hiseq_img.save(
                '{}/hiseq_{}/{}.png'.format(wctx.out_path, sizes[1], i.zfill(5)))
        else:
            with wctx.env.begin(write=True) as txn:
                txn.put('lr_{}_{}'.format(
                    wctx.sizes[0], i.zfill(5)).encode('utf-8'), lr_img)
                txn.put('hr_{}_{}'.format(
                    wctx.sizes[1], i.zfill(5)).encode('utf-8'), hr_img)
                txn.put('sr_{}_{}_{}'.format(
                    wctx.sizes[0], wctx.sizes[1], i.zfill(5)).encode('utf-8'), sr_img)
                txn.put('hiseq_{}_{}'.format(
                    wctx.sizes[1], i.zfill(5)).encode('utf-8'), hiseq_img)
        curr_total = wctx.inc_get()
        if wctx.lmdb_save:
            with wctx.env.begin(write=True) as txn:
                txn.put('length'.encode('utf-8'), str(curr_total).encode('utf-8'))

def all_threads_inactive(worker_threads):
    for thread in worker_threads:
        if thread.is_alive():
            return False
    return True

def prepare(img_path, out_path, n_worker, sizes=(16, 128), resample=Image.BICUBIC, 
                lmdb_save=False, gamma_range=(1.5, 1.7)):
    resize_fn = partial(resize_worker, sizes=sizes,
                        resample=resample, lmdb_save=lmdb_save,
                        gamma_range=gamma_range)
    files = [p for p in Path(
        '{}'.format(img_path)).glob(f'**/*')]

    if not lmdb_save:
        os.makedirs(out_path, exist_ok=True)
        os.makedirs('{}/lr_{}'.format(out_path, sizes[0]), exist_ok=True)
        os.makedirs('{}/hr_{}'.format(out_path, sizes[1]), exist_ok=True)
        os.makedirs('{}/sr_{}_{}'.format(out_path,
                    sizes[0], sizes[1]), exist_ok=True)
        os.makedirs('{}/hiseq_{}'.format(out_path, sizes[1]), exist_ok=True)
    else:
        env = lmdb.open(out_path, map_size=1024 ** 4, readahead=False)

    if n_worker > 1:
        # prepare data subsets
        multi_env = None
        if lmdb_save:
            multi_env = env

        file_subsets = np.array_split(files, n_worker)
        worker_threads = []
        wctx = WorkingContext(resize_fn, lmdb_save, out_path, multi_env, sizes)

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
            if not lmdb_save:
                lr_img.save(
                    '{}/lr_{}/{}.png'.format(out_path, sizes[0], i.zfill(5)))
                hr_img.save(
                    '{}/hr_{}/{}.png'.format(out_path, sizes[1], i.zfill(5)))
                sr_img.save(
                    '{}/sr_{}_{}/{}.png'.format(out_path, sizes[0], sizes[1], i.zfill(5)))
                hiseq_img.save(
                    '{}/hiseq_{}/{}.png'.format(out_path, sizes[1], i.zfill(5)))
            else:
                with env.begin(write=True) as txn:
                    txn.put('lr_{}_{}'.format(
                        sizes[0], i.zfill(5)).encode('utf-8'), lr_img)
                    txn.put('hr_{}_{}'.format(
                        sizes[1], i.zfill(5)).encode('utf-8'), hr_img)
                    txn.put('sr_{}_{}_{}'.format(
                        sizes[0], sizes[1], i.zfill(5)).encode('utf-8'), sr_img)
                    txn.put('hiseq_{}_{}'.format(
                        sizes[1], i.zfill(5)).encode('utf-8'), hiseq_img)
            total += 1
            if lmdb_save:
                with env.begin(write=True) as txn:
                    txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str,
                        default='/home/allen/Documents/MIE288/sr3_server4/dataset/dark/temp/lr_400'.format(Path.home()))
    parser.add_argument('--out', '-o', type=str,
                        default='./dataset/dark/LOLdataset')
    parser.add_argument('--size', type=str, default='400,400')
    parser.add_argument('--n_worker', type=int, default=20)
    parser.add_argument('--resample', type=str, default='bicubic')
    # default save in png format
    parser.add_argument('--lmdb', '-l', action='store_true')
    # low-light gamma division (larger number, darker img)
    parser.add_argument('--gamma', '-g', type=str, default='1.5,1.7')

    args = parser.parse_args()

    resample_map = {'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
    resample = resample_map[args.resample]
    sizes = [int(s.strip()) for s in args.size.split(',')]
    gamma_range = [float(s.strip()) for s in args.gamma.split(',')]

    args.out = '{}_{}_{}_gamma_{}_{}'.format(args.out, sizes[0], sizes[1], 
                                             gamma_range[0], gamma_range[1])
    prepare(args.path, args.out, args.n_worker,
            sizes=sizes, resample=resample, lmdb_save=args.lmdb, gamma_range=gamma_range)
