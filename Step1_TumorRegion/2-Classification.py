import os
import glob
import fastai
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix
from openslide import OpenSlide
from fastai.vision.all import *
from sklearn.utils import shuffle


def load_image(fn, mode=None):
    slide_path, h, w, level, ps = fn.split('&')
    h = int(h); w = int(w)
    level = int(level); ps = int(ps)
    
    with OpenSlide(slide_path) as slide:
        im = slide.read_region((w, h), level, (ps, ps)).convert('RGB')
    
    im.load()
    im = im._new(im.im)
    return im.convert(mode) if mode else im

fastai.vision.core.load_image = load_image

def get_df(csv_list, num=100):
    df = []

    for item in csv_list:
        data = pd.read_csv(item)
        pos = shuffle(data[data['label'] == 1])
        neg = shuffle(data[data['label'] == 0])
        df_sample = pd.concat([pos[:num], neg[:num]])

        df += [df_sample]

    df = pd.concat(df)
    
    return df

train_csv = glob.glob('**/train/*.csv')
train_df = get_df(train_csv)
train_df['is_valid'] = 0

eval_csv = glob.glob('**/eval/*.csv')
eval_df = get_df(eval_csv)
eval_df['is_valid'] = 1

df = pd.concat([train_df, eval_df])

path = '/'

dls = ImageDataLoaders.from_df(
    df, 
    path,
    valid_col='is_valid',
    item_tfms=Resize(256),
    batch_tfms=aug_transforms(size=256)
)

dls.show_batch()

learn = vision_learner(
    dls, 
    resnet18,
    metrics=[accuracy],
)

learn.model_dir = '**/Step1_TumorRegion/dataset/model'

learn.fine_tune(
    12,
    base_lr = 3e-4,
    cbs=[SaveModelCallback(fname='best')],
)

learn.load('best')

p, t = learn.get_preds()
p = p.numpy()
t = t.numpy()
p = np.argmax(p, axis=1)

confusion_matrix(t, p)

csv_list = glob.glob('**/test/*.csv')
test_df = []

for item in csv_list:
    data = pd.read_csv(item)
    pos = shuffle(data[data['label'] == 1])
    neg = shuffle(data[data['label'] == 0])
    df_sample = pd.concat([pos, neg])

    test_df += [df_sample]

test_df = pd.concat(test_df)

test_dl = learn.dls.test_dl(test_df)

p, _ = learn.get_preds(dl=test_dl)
p = p.numpy()
p_oh = np.argmax(p, axis=1)

confusion_matrix(test_df['label'].tolist(), p_oh)

np.sum(test_df['label'].tolist() == p_oh) / len(p_oh)

test_df['pred'] = p[:, 1]
test_df.to_csv('**/preds/test_pred.csv', index=None)