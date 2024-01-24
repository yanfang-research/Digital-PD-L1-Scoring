import os
import cv2
import glob
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm.notebook import tqdm
from openslide import OpenSlide
from multiprocessing import Pool


slide_base_path = '**/IHC'

def load_image(fn, mode=None):
    fn = fn.split('/')[-1][:-4]
    
    slide_name, h, w, level, ps = fn.split('&')
    h = int(h); w = int(w)
    level = int(level); ps = int(ps)
    slide_path = '{}/{}'.format(slide_base_path, slide_name)
    
    with OpenSlide(slide_path) as slide:
        im = slide.read_region((w, h), level, (ps, ps)).convert('RGB')
    
    return im

def dilate(mask):
    mask = np.array(mask, np.uint8)
    kernel = np.ones((9, 9),np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    
    return mask != 0

def get_pos_mask(img_rgb):
    try:
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV) 
    except:
        print(img_rgb.shape)
    
    lower_red = np.array([0, 43, 46]) 
    upper_red = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    return np.array(mask != 0, np.uint8)

def get_area_prec(mask_path, is_dilate=True):
    img = np.array(load_image(mask_path))
    mask = np.array(Image.open(mask_path))
    exclude_idx = set(mask[:, [0, -1]].flatten()).union(set(mask[[0, -1], :].flatten()))
    _max = np.max(mask)

    area_list = []
    pos_prec_list = []
    
    for i in range(1, _max+1):
        if i in exclude_idx:
            continue

        cell_mask = mask == i
        
        if is_dilate:
            cell_mask = dilate(cell_mask)
        
        area = np.sum(cell_mask)
        
        try:
            pos_prec = np.mean(get_pos_mask(img[cell_mask][np.newaxis, :, :]))
            area_list += [area]
            pos_prec_list += [pos_prec]
        except:
            print(i, mask_path)
            return [],[]
        
    return area_list, pos_prec_list

def get_wsi_sort(wsi_name):
    name_img_list = img_list[name_list == wsi_name]
    
    with Pool(processes=16) as p:
        results = p.map(get_area_prec, name_img_list)
    
    area_list = np.concatenate([item[0] for item in results])
    pos_prec_list = np.concatenate([item[1] for item in results])
    
    return area_list, pos_prec_list


img_list = np.array(glob.glob('**/preds/*.png'))
name_list = np.array([item.split('/')[-1].split('.')[0] for item in img_list])
name_sets = set(name_list)

for name in tqdm(name_sets):
    area_list, pos_prec_list = get_wsi_sort(name)
    df = pd.DataFrame()
    df['area'] = area_list
    df['pos'] = pos_prec_list
    df.to_csv('../dataset/cell_area_pos/{}.csv'.format(name), index=None)
