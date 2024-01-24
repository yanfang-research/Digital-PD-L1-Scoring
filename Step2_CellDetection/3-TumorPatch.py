import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns

from math import ceil
from PIL import Image
from tqdm.notebook import tqdm
from pathadox_allslide.AllSlide import AllSlide


def save_infer_df(thres=0.4, num=512):    
    test_pred = pd.read_csv(test_pred_path)
    tumor_df = test_pred[test_pred['pred'] > thres]
    df_num = ceil(len(tumor_df) / num)

    for i in range(df_num):
        s = i * num
        e = (i + 1) * num if i != df_num-1 else len(tumor_df)
        i_df = tumor_df[s:e]
        save_path = '{}/{}.csv'.format(infer_df_path, i)
        i_df.to_csv(save_path, index=None)


test_pred_path = '**/Step1_TumorRegion/dataset/preds/test_pred.csv'
infer_df_path = '**/Step2_CellDetection/dataset/infer_df'

save_infer_df()