from PIL import Image
import os
from tqdm import tqdm
import numpy as np
folder = 'X'
all_images = os.listdir(f'./Slices{folder}/')
all_images.sort()

start_idx = 80
end_idx = 1600

for i, img_name in enumerate(all_images):
    if('tif' not in img_name):
        continue
    img_prefix = img_name.split('.')[0]
    img_idx = int(img_name.split('.')[0].replace('slice', ''))
    if(img_idx < start_idx or img_idx > end_idx):
        continue
    if(img_idx % 15 != 0):
        continue
    img = Image.open(os.path.join(f'./Slices{folder}/',img_name))#.convert('RGB')
    
    img_name_png = f"{img_name.split('.')[0]}.png"
    img.save(os.path.join(f'./Slice{folder}_png/',img_name_png))

    img = Image.open(os.path.join(f'./Slice{folder}_png/',f"{img_prefix}.png"))
    img_array = np.array(img)
    h,w = img_array.shape
    img_array = img_array[:, 50:w-50]
    img_save = Image.fromarray(img_array)
    img_save.save(os.path.join(f'./Slice{folder}_png/',img_name_png))
