import os
import cv2
import glob
import json
import shutil
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from PIL import Image
from multiprocessing import Pool
from skimage.feature import peak_local_max


def json2dots(json_path, target_size=256):
    base_name = json_path.split('/')[-1].split('.')[0]
    with open(json_path) as f:
        data = json.load(f)
    
    h = data['imageHeight']
    ds_ratio = int(h / target_size)
    
    points = np.array([item['points'][0] for item in data['shapes']], np.int16)
    points = np.array(points/ds_ratio, np.int16)
    
    dots_map = np.zeros((target_size, target_size))
    dots_map[points[:, 1], points[:, 0]] = 255
    dots_map = np.array(dots_map, np.uint8)
    
    Image.fromarray(dots_map).save('{}/{}.png'.format(dot_path, base_name))
    
def split_one_sample(img_path, target_size=256):
    base_name = img_path.split('/')[-1].split('_')[0]
    
    if base_name in train_names:
        _type = 'train'
    elif base_name in eval_names:
        _type = 'eval'
    elif base_name in test_names:
        _type = 'test'
    else:
        print('Error!')
        
    dot_path = img_path.replace('/images/', '/dots/').replace('.jpg', '.png')
    dot_save_path = '{}/{}/'.format(data_path, _type)
    shutil.copy(dot_path, dot_save_path)
    
    img_save_path = '{}/{}/{}'.format(data_path, _type, img_path.split('/')[-1])
    Image.open(img_path).resize((target_size, target_size)).save(img_save_path)
    

dot_path = '../dataset/dots'
img_path = '../dataset/images'
json_path = '../dataset/jsons'
data_path = '../dataset/data'
step1_split_data_path = '../../Step1_TumorRegion/dataset/data'

json_list = glob.glob('{}/*.json'.format(json_path)); len(json_list)

for item in json_list:
    json2dots(item)
    

train_names = [item.split('/')[-1].split('.')[0] for item in glob.glob('{}/train/*.csv'.format(step1_split_data_path))]
eval_names = [item.split('/')[-1].split('.')[0] for item in glob.glob('{}/eval/*.csv'.format(step1_split_data_path))]
test_names = [item.split('/')[-1].split('.')[0] for item in glob.glob('{}/test/*.csv'.format(step1_split_data_path))]

img_list = glob.glob('{}/*.jpg'.format(img_path)); len(img_list)

for img_path in img_list:
    split_one_sample(img_path)

    
