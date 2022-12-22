from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import cv2

# img = cv2.imread('slice00105.png')
# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# output_hsv = img_hsv.copy()
# output_hsv[np.where(img_hsv[:,:,2]>40)] = 255
# output_hsv[np.where(img_hsv[:,:,2]<=40)] = 0
# cv2.imwrite('test.png', output_hsv)
red_label_folder = 'ourdata/SliceY_png_with_mark'
png_folder = 'ourdata/SliceY_png_original'
full_mask_folder = 'ourdata/SliceY_mask'
if('SliceX' in red_label_folder):
    direction = 'X'
if('SliceY' in red_label_folder):
    direction = 'Y'
if('SliceZ' in red_label_folder):
    direction = 'Z'

save_img_folder = 'ourdata/images'
mask_folder = 'ourdata/masks'
imgs = os.listdir(red_label_folder)
imgs.sort()

for img_name in imgs:
    img_name = img_name.split('.')[0]
    new_img_name = f'{img_name}_{direction}'
    img = Image.open(os.path.join(red_label_folder, f'{img_name}.png')).convert('RGB')
    img = np.array(img)
    max_p, min_p = np.max(img), np.min(img)
    if(max_p == min_p and min_p == 255):
        img = img / 65535.0 * 255.0
    # print(img[0,1])
    h,w,_ = img.shape
    img_mask = np.zeros((h,w,3))
    labeled = False
    for i in range(h):
        for j in range(w):
            r,g,b = img[i,j]
            if(r != g):
                labeled = True
                img_mask[i,j, :] = 255
    if(labeled):
        img_mask_pil = Image.fromarray(img_mask.astype(np.uint8)).convert('L')
        img_mask_pil.save(os.path.join(full_mask_folder, f'{new_img_name}.png'))

        ori_img = Image.open(os.path.join(png_folder, f'{img_name}.png'))#.convert('RGB')
        ori_img = np.array(ori_img)
        ori_img = ori_img / 65535.0 * 255.0

        if(h > w):
            cut_num = h // w
            for cut_idx in range(cut_num):
                cur_img = ori_img[cut_idx*w:(cut_idx+1)*w, :]
                cur_mask = img_mask[cut_idx*w:(cut_idx+1)*w, :, :]
                cur_img = Image.fromarray(cur_img.astype(np.uint8)).convert('RGB')


                cur_img.save(os.path.join(save_img_folder, f'{new_img_name}_{cut_idx}.png'))

                cur_mask = Image.fromarray(cur_mask.astype(np.uint8)).convert('L')
                cur_mask.save(os.path.join(mask_folder, f'{new_img_name}_{cut_idx}.png'))
        else:# this is for Y
            ori_img = ori_img[50:h-50, :]
            img_mask = img_mask[50:h-50, :]
            h, w= ori_img.shape
            cut_num = w // h
            for cut_idx in range(cut_num):

                cur_img = ori_img[:, cut_idx*h:(cut_idx+1)*h]
                cur_mask = img_mask[:, cut_idx*h:(cut_idx+1)*h, :]

                cur_img = Image.fromarray(cur_img.astype(np.uint8)).convert('RGB')

                cur_img.save(os.path.join(save_img_folder, f'{new_img_name}_{cut_idx}.png'))

                cur_mask = Image.fromarray(cur_mask.astype(np.uint8))
                cur_mask.save(os.path.join(mask_folder, f'{new_img_name}_{cut_idx}.png'))


# if __name__ == "__main__":
#     imga = Image.open('/Users/zhangyuyao/Desktop/Exchange EPFL/ml-project-2-cracksegfrp/ourdata/SliceY_png_original/slice00015.png').convert('RGB')
#     imga = np.array(imga)
#     h, w, _  = imga.shape
#     count = 0
#     c = 0
#     for i in range(h):
#         for j in range(w):
#             c += 1
#             r,g,b = imga[i,j]
#             if r==g and r==b:
#                 count += 1
#     print(c)
#     print(count)
