from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torch
from torchvision.transforms import functional as T

# img_path = '/home/allen/Documents/MIE288/sr3_server4/data/dark/jenny/test_512/ir_512'
# out_path = '/home/allen/Documents/MIE288/sr3_server4/data/dark/jenny_test_256_256/ir'
# size = 256

# files = [p for p in Path('{}'.format(img_path)).glob(f'**/*')]

# for file in tqdm(files):
#     img = Image.open(file).convert('RGB')
#     img = T.resize(img, size, Image.BICUBIC)

#     filename = file.name.split('.')[0]
#     filename = filename[-8:-5]

#     img.save(f'{out_path}/{filename.zfill(5)}.png')

#     # tensor = T.to_tensor(img)

#     # k=torch.sum(tensor[0]-tensor[1])
#     # print()

bchw = torch.randn(size=[1,1,256,256])
b3chw = bchw.repeat(1,3,1,1)

print(bchw.shape)
print(b3chw.shape)