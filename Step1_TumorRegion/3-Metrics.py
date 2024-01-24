import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
from pathadox_allslide.AllSlide import AllSlide


def dice(pd, gt, threshold=0.5, eps=1e-8):
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


tumor_mask_path = '**/tumor'
pred_path = '**/preds'
svs_path = '**/IHC'

results = np.array(pd.read_csv('{}/test_pred.csv'.format(pred_path)))
auc = roc_auc_score(np.array(results[:, 1], np.float), np.array(results[:, 2], np.float))