import cv2
import numpy as np
import os

####################################################################
#degradate the face image: code generated by Jenny on 10 Aug 2023
#degradate the blue channel
#Gamma degradation
#add Gaussian Noise
####################################################################
def creat_gamma_lut(gamma):
    lut = np.array([((i/255.0)**gamma)*255 for i in range(256)])
    lut = np.clip(lut, 0, 255).astype((np.uint8))
    return lut


def image_degradation(image, degradation_image_name):
    image_blue_degradation = image.copy()
    #blue channel degradation, factor range from 0.6 to 0.8
    min_factor = 0.6
    max_factor = 0.8
    blue_degradatoin_factor = np.random.uniform(min_factor,max_factor)
    print(blue_degradatoin_factor)
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
    gamma=np.random.uniform(1.0,1.2)
    lut = creat_gamma_lut(gamma)
    gamma_image = cv2.LUT(noisy_image,lut)


    #jpeg degradation of the image, range is 50~100
    jpeg_quality = np.random.randint(40,95)
    jpeg_compression_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    cv2.imwrite(degradation_image_name, gamma_image, params=jpeg_compression_params)

    '''    
    compressed_image = cv2.imread(degradation_image_name)
    cv2.imshow("blue degradation", image_blue_degradation)
    cv2.imshow("contrast_degradation", contrast_brightness_adjust_image)
    cv2.imshow("Gaussian_degradatoin", noisy_image)
    cv2.imshow("gamma_degradation", gamma_image)
    cv2.imshow("jpeg", compressed_image)
    cv2.imshow("original", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''


def file_name(file_dir):
   L=[]
   len = 0
   for root, dirs, files in os.walk(file_dir):
      files.sort()#key=lambda x: int(x[4:]))
      for file in files:
         if os.path.splitext(file)[1] == '.png':  # 想要保存的文件格式
            L.append(os.path.join(root, file))
   return L

file_name_list = file_name("/home/icduser/disk_c/Jenny/ARD288/dataset/archive_Flickr_faces_HQ/")
out_folder ='/home/icduser/disk_c/Jenny/ARD288/dataset/archive_Flickr_faces_HQ_dark/'
if (os.path.exists(out_folder) == False):
    os.makedirs(out_folder)
for image_file_name in file_name_list:
    image =  cv2.imread(image_file_name)
    degradation_image_name = os.path.splitext(os.path.split(image_file_name)[1])[0]
    degradation_image_name = out_folder+degradation_image_name +'_dark.jpg'
    image_degradation(image, degradation_image_name)
