import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from skimage.feature import peak_local_max


def norm_0_1(mask):
    x_min = np.min(mask)
    x_max = np.max(mask)
    
    new_mask = (mask-x_min)/(x_max-x_min)
    
    return new_mask

def get_cell_points(den_map, min_dis=2, thres=0.4):
    
    if np.max(den_map) < thres:
        return []
    
    den_map = norm_0_1(den_map)
    
    x_y = peak_local_max(
        den_map, 
        min_distance = min_dis,
        threshold_abs = thres,
    )
    
    return x_y

def get_points_mask(den_map):
    x_y = get_cell_points(den_map)
    mask = np.zeros_like(den_map)
    
    mask[x_y[:, 0], x_y[:, 1]] = 1.
    
    return mask

def MAE(pred, label):
    pred[pred != 0] = 1
    label[label != 0] = 1
    
    score = np.abs(np.sum(pred) - np.sum(label))
    return score

def MACE(pred, label):
    label[label != 0] = 1
    
    score = np.abs(np.sum(pred)/100. - np.sum(label))
    return score

def GAME(pred, label, n=4):
    pred[pred != 0] = 1
    label[label != 0] = 1
    
    pred = np.concatenate([np.split(item, n, axis=0) for item in np.split(pred, n, axis=1)])
    label = np.concatenate([np.split(item, n, axis=0) for item in np.split(label, n, axis=1)])
    
    score = []
    for i in range(n*n):
        score += [MAE(pred[i], label[i])]
    
    return np.mean(score)

def get_metrics(json_path):
    with open(json_path) as f:
        data = json.load(f)
        
    pred = np.array(data['pred'])
    img_list = np.array(data['img_path'])
    
    for idx in range(len(pred)):
        
        name = img_list[idx].split('/')[-1].split('.')[0]
        
        pred_density = pred[idx]
        pred_mask = get_points_mask(pred[idx])
        label_mask = np.array(Image.open(img_path.format(name)))
        
    mae = MAE(pred_mask, label_mask)
    mace = MACE(pred_density, label_mask)
    game_16 = GAME(pred_mask, label_mask, n=4)
    game_64 = GAME(pred_mask, label_mask, n=8)
    
    print('MAE: {}, MACE: {}, GAME-16: {}, GAME-64: {}'.format(mae, mace, game_16, game_64))
    

json_path = '**/Step2_CellDetection/code/lightning_logs/PDL1/version_0/result.json'
img_path = '**/Step2_CellDetection/dataset/data/test/{}.png'

get_metrics(json_path)