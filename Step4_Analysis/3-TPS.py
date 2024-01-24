import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns

from functools import partial
from tqdm.notebook import tqdm
from multiprocessing import Pool


def get_pos_score(csv_path, prec, pos_thres, num):
    data = pd.read_csv(csv_path).sort_values('area')
    e_num = len(data) - int(len(data) * prec)
    score_ar = np.array(data['pos'].tolist())
    top_score = np.array(score_ar[e_num-num:e_num])
    tps = np.sum(top_score > pos_thres) / len(top_score)
    
    return tps

def get_all_pos_score(csv_list, prec=0.02, pos_thres=0.1, num=1000):
    func = partial(get_pos_score, prec=prec, pos_thres=pos_thres, num=num)
    
    with Pool(processes=16) as p:
        tps_score = p.map(func, csv_list)
    
    return tps_score

cell_number = []
for item in total_data['sdpc']:
    try:
        cell_number += [pd.read_csv('../dataset/cell_area_pos/{}.csv'.format(item)).shape[0]]
    except:
        cell_number += [-1]


# Calculate TPS

total_data = pd.read_csv('../dataset/TPS_Survival.csv').fillna(-1)
total_data['cell_number'] = cell_number

total_data = total_data[
    (total_data['tps1'] != -1) & (total_data['tps2'] != -1) & (total_data['tps3'] != -1) & (total_data['cell_number'] > 5000)
]
sdpc_list = total_data['sdpc'].tolist()

tps_d1 = total_data['tps1'].tolist()
tps_d2 = total_data['tps2'].tolist()
tps_d3 = total_data['tps3'].tolist()
csv_list = ['../dataset/cell_area_pos/{}.csv'.format(item) for item in sdpc_list]; len(csv_list)

# Total

prec_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
pos_thres_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
num_list = [4000]

df = pd.DataFrame()
df['sdpc'] = sdpc_list
df['tps1'] = tps_d1
df['tps2'] = tps_d2
df['tps3'] = tps_d3

for prec in prec_list:
    for pos_thres in pos_thres_list:
        for num in num_list:
            _str = '{}_{}_{}'.format(str(prec), str(pos_thres), str(num))
            print(_str)
            pos_list = get_all_pos_score(csv_list, prec, pos_thres, num)
            df[_str] = pos_list
            
mean = np.mean(df.iloc[:, 1:4], axis=1).tolist()
median = np.median(df.iloc[:, 1:4], axis=1).tolist()

df['mean'] = mean
df['median'] = median

df.to_csv('./results.csv', index=None)

# Evaluation

def mae(a, b):
    score = np.mean(np.abs(a - b))
    return score

def cac_mae(df, num):
    prec_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    pos_thres_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    score = np.zeros((6, 6))
    
    for idx1, pos in enumerate(pos_thres_list):
        for idx2, prec in enumerate(prec_list):
            column_name = '{}_{}_{}'.format(prec, pos, num)
            
            x = np.array(df[column_name].tolist()) * 100.
            y = df['median']
            
            score[idx1, idx2] = mae(x, y)
    
    return score


df = pd.read_csv('./results.csv')

score = cac_mae(df, num=4000)
pd.DataFrame(score, index=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5], columns=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06])











