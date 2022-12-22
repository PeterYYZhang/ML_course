from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import cv2

def mosaic(img_list, mask_list):
    assert len(img_list) == 4
    h,w,c = img_list[0].shape
    mosaic_img = np.zeros((h,w,c))
    mosaic_img_mask = np.zeros((h,w))
    s_h = np.random.randint(low = h // 4, high = h // 4 * 3)
    s_w = np.random.randint(low = w // 4, high = w // 4 * 3)
    coord_start = [(0, 0), (0, s_w), (s_h, 0), (s_h, s_w)]
    coord_end = [(s_h, s_w), (s_h, w), (h, s_w), (h, w)]
    for i in range(len(img_list)):
        cur_img = img_list[i]
        cur_mask = mask_list[i]
        start_h = coord_start[i][0]
        end_h = coord_end[i][0]
        start_w = coord_start[i][1]
        end_w = coord_end[i][1]

        mosaic_img[start_h:end_h, start_w:end_w, :] = cur_img[start_h:end_h, start_w:end_w, :]
        mosaic_img_mask[start_h:end_h, start_w:end_w] = cur_mask[start_h:end_h, start_w:end_w]
    return mosaic_img, mosaic_img_mask

image_folder = 'images'
mask_folder = 'masks'
mosaic_folder = 'mosaic'
mosaic_mask_folder = 'mosaic_mask'
imgs = os.listdir(image_folder)
imgs.sort()
mosaic_img_num = 2
for img_num in range(mosaic_img_num):
    img_names = np.random.choice(imgs, size = 4, replace = False)
    img_list = []
    mask_list = []
    for name in img_names:
        img = np.array(Image.open(os.path.join(image_folder, name)))
        mask = np.array(Image.open(os.path.join(mask_folder, name)))
        img_list.append(img)
        mask_list.append(mask)
    mosaic_img, mosaic_mask = mosaic(img_list, mask_list)
    mosaic_img = Image.fromarray(mosaic_img.astype(np.uint8)).convert('RGB')
    mosaic_mask = Image.fromarray(mosaic_mask.astype(np.uint8)).convert('L')
    mosaic_img.save(os.path.join(mosaic_folder, f'{img_num}.png'))
    mosaic_mask.save(os.path.join(mosaic_mask_folder, f'{img_num}.png'))
