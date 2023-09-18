from PIL import Image

img_path = '/home/allen/Documents/MIE288/sr3_server4/data/dark/ffhq_512_512_gamma_1_1.2/hr_512'

img = Image.open(img_path).convert("RGB")

sz = img.size

print()