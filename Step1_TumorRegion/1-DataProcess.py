import os
import xml
import cv2
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from xml.etree.ElementTree import parse
from openslide import OpenSlide
from PIL import ImageEnhance
from scipy import ndimage
from sklearn.utils import shuffle
from tqdm.notebook import tqdm
from multiprocessing import Pool
from functools import partial
from skimage.morphology import remove_small_objects


def create_tissue_mask(slide, level=2):
    col, row = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (col, row))
    
    tissue_mask = np.array(
        ImageEnhance.Contrast(img).enhance(4).convert('L')
    )

    tissue_mask = np.array(tissue_mask < 200., np.uint8)

    tissue_mask = cv2.blur(tissue_mask, (12, 12))
    tissue_mask = np.array(ndimage.binary_fill_holes(tissue_mask), np.uint8)
    
    return tissue_mask

def create_xml_mask(annotation, slide, level=2):
    col, row = slide.level_dimensions[level]
    mask = np.zeros((row, col))

    for i in range(len(annotation)):
        mask_temp = np.zeros((row, col))
        cv2.drawContours(mask_temp, [annotation[i]], -1, 255, -1)
        mask[mask_temp != 0] = 1
        
    return mask

def get_annotation_from_xml(target_xml_path, downsamples=16):
    annotation = []
    num_annotation = 0

    tree = parse(target_xml_path)
    root = tree.getroot()
    for Annotation in root.iter("Annotation"):
        annotation_list = []
        for Coordinate in Annotation.iter("Coordinate"):
            x = round(float(Coordinate.attrib["X"]) / downsamples)
            y = round(float(Coordinate.attrib["Y"]) / downsamples)
            annotation_list.append((x, y))
        annotation.append(np.asarray(annotation_list))

    return annotation

def extract_tumor_non_tumor_df(non_tumor_xml_path, ps=256, remove_size=16, num=100):
    base_name = non_tumor_xml_path.split('/')[-1].split('.')[0]
    slide_path = '{}/{}.svs'.format(svs_path, base_name)
    roi_xml_path = '{}/{}.xml'.format(roi_mask_path, base_name)
    
    slide = OpenSlide(slide_path)
    tissue_mask = create_tissue_mask(slide)
    rs_size = np.int64(np.array(slide.level_dimensions[0]) / ps)
    
    roi_annotation = get_annotation_from_xml(roi_xml_path)
    roi_mask = create_xml_mask(roi_annotation, slide)
    roi_mask = np.array((roi_mask != 0) & (tissue_mask != 0), np.uint8)
    roi_mask = cv2.resize(roi_mask, rs_size)
    
    non_tumor_annotation = get_annotation_from_xml(non_tumor_xml_path)
    non_tumor_mask = create_xml_mask(non_tumor_annotation, slide)
    non_tumor_mask = np.array((non_tumor_mask != 0) & (tissue_mask != 0), np.uint8)
    non_tumor_mask = cv2.resize(non_tumor_mask, rs_size)
    non_tumor_mask = np.array((non_tumor_mask != 0) & (roi_mask != 0), np.uint8)
    
    tumor_mask = np.array((roi_mask != 0) & (non_tumor_mask != 1), np.uint8)
    
    Image.fromarray(tumor_mask).save('{}/tumor/{}.png'.format(mask_save_path, base_name))
    Image.fromarray(non_tumor_mask).save('{}/non_tumor/{}.png'.format(mask_save_path, base_name))
    
def get_df(tumor_path, _type, ps=256):
    base_name = tumor_path.split('/')[-1].split('.')[0]
    non_tumor_path = tumor_path.replace('/tumor/', '/non_tumor/')

    slide_path = '{}/{}.svs'.format(svs_path, tumor_path.split('/')[-1].split('.')[0])

    tumor_mask = np.array(Image.open(tumor_path))
    non_tumor_mask = np.array(Image.open(non_tumor_path))

    t_h, t_w = np.where(tumor_mask != 0)
    t_h_s = t_h * ps; t_w_s = t_w * ps
    tumor_list = ['{}&{}&{}&{}&{}'.format(slide_path, str(item[0]), str(item[1]), str(0), str(ps)) for item in zip(t_h_s, t_w_s)]

    u_h, u_w = np.where(non_tumor_mask != 0)
    u_h_s = u_h * ps; u_w_s = u_w * ps
    non_tumor_list = ['{}&{}&{}&{}&{}'.format(slide_path, str(item[0]), str(item[1]), str(0), str(ps)) for item in zip(u_h_s, u_w_s)]

    df = pd.DataFrame()
    df['img'] = tumor_list + non_tumor_list
    df['label'] = [1] * len(tumor_list) + [0] * len(non_tumor_list)
    
    df.to_csv('{}/{}/{}.csv'.format(df_path, _type, base_name), index=None)
    

svs_path = '**/IHC'
mask_save_path = '**/mask'
df_path = '**/data'
roi_mask_path = '**/roi'

non_tumor_xml_list = glob.glob('**/non_tumor/*.xml', recursive=True)
len(non_tumor_xml_list)

with Pool(processes=8) as p:
    p.map(extract_tumor_non_tumor_df, non_tumor_xml_list)
    
tumor_list = glob.glob('.**/tumor/*.png', recursive=True)
tumor_list.sort()
num = int(len(tumor_list) / 5)

p1 = tumor_list[:num]
p2 = tumor_list[num:2*num]
p3 = tumor_list[2*num:3*num]
p4 = tumor_list[3*num:4*num]
p5 = tumor_list[4*num:]

tumor_train = p1 + p2 + p3
tumor_valid = p4
tumor_test = p5

for item in tumor_train:
    get_df(item, 'train')
    
for item in tumor_valid:
    get_df(item, 'eval')
    
for item in tumor_test:
    get_df(item, 'test')