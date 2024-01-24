import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from pathadox_allslide.AllSlide import AllSlide


def iou(pd, gt, threshold, eps=1e-8):
    pd = np.array(pd > threshold, np.uint8)
    
    _sum = pd + gt
    inter = np.sum(_sum == 2) + eps
    union = np.sum(_sum != 0) + eps
    
    return inter / union

def dice(pd, gt, threshold, eps=1e-8):
    pd = np.array(pd > threshold, np.uint8)
    
    inter = np.sum((pd == 1) & (gt == 1))
    union = np.sum(pd) + np.sum(gt) + eps
    
    return 2 * inter / union

def get_heatmap(base_name, results):
    tumor_path = '{}/{}.png'.format(tumor_mask_path, base_name)
    tumor_mask =  np.array(Image.open(tumor_path))
    name_list = np.array([item.split('.')[0].split('/')[-1] for item in results[:, 0]])
    idx = name_list == base_name
    select_df = results[idx]
    pred_mask = np.zeros_like(tumor_mask)
    pred_mask = np.array(pred_mask, np.float16)

    for fn, _, score in select_df:
        slide_path, h, w, level, ps = fn.split('&')
        h = int(h); w = int(w)
        level = int(level); ps = int(ps)
        h = int(h/ps); w=int(w/ps)

        pred_mask[h, w] = score
    
    pred_mask = np.array(pred_mask, np.float)
    
    return pred_mask, tumor_mask

def get_iou_dice(results, thres):
    name_sets = set(np.array([item.split('.')[0].split('/')[-1] for item in results[:, 0]]))
    
    dice_score = []
    iou_score = []
    
    for base_name in name_sets:
        heatmap, tumormask = get_heatmap(base_name, results)
        dice_score += [dice(heatmap, tumormask, thres)]
        iou_score += [iou(heatmap, tumormask, thres)]
        
    return np.mean(dice_score), np.mean(iou_score)

def get_metrics(results, thres=0.4):
    t = np.array(results[:, 1], np.float32)
    p = np.array(results[:, 2], np.float32)
    p_oh = np.array(p > thres)
    
    auc = roc_auc_score(t, p)
    acc = accuracy_score(t, p_oh)
    f1 = f1_score(t, p_oh)
    recall = recall_score(t, p_oh)
    precision = precision_score(t, p_oh)
    dice, iou = get_iou_dice(results, thres)
    
    print('ACC: {}, AUC: {}, F1: {}, Recall: {}, Precision: {}, IoU: {}, Dice: {}.'.format(
        acc,
        auc,
        f1,
        recall,
        precision,
        iou,
        dice
    ))
    
    print('Confusion Matrix: \n', confusion_matrix(t, p_oh))
    

tumor_mask_path = '.**/mask/tumor'
pred_path = '**/preds'
svs_path = '**/IHC'

results = np.array(pd.read_csv('{}/test_pred.csv'.format(pred_path)))
get_metrics(results)