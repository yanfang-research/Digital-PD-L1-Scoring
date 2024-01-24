import os
import cv2
import json
import glob
import torch
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

from PIL import Image
from multiprocessing import Pool
from skimage.feature import peak_local_max
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl

from core.model import *
from core.data import *
from core.metrics import *


DATASETS = 'PDL1'
GPUS = 1
SIGMA = 3
BS = 16
RF = 0.9
NW = 4
WD = 1e-5
LR = 3e-4
DIM = 256
EPOCHS = 1000
IN_CHANNELS = 3
SAVE_TOP_K = 5
EARLY_STOP = 20
EVERY_N_EPOCHS = 1
LOG_EVERY_N_STEPS = 1
WEIGHTS = torch.FloatTensor([1./64, 1./16., 1./4, 1.])
DF_PATH = '../dataset/infer_df'
SAVE_PATH = '../dataset/infer_df_pred'
ckpt_path = '../code/lightning_logs/PDL1/version_0/checkpoints/epoch=91-val_mce=9.17.ckpt'

# pytorch lightning module

class FCRN(pl.LightningModule):
    def __init__(self, in_channels):
        super().__init__()
        self.fcrn = C_FCRN_Aux(in_channels)
        self.loss = MyLoss(WEIGHTS)

    def forward(self, x):
        out = self.fcrn(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LR, weight_decay=WD)
        scheduler = ReduceLROnPlateau(optimizer, factor=RF, mode='min', patience=4, min_lr=0, verbose=True)
        # patience=10
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'val_mce'
        }

    def training_step(self, train_batch, batch_idx):
        x, d1, d2, d3, d4 = train_batch
        y = [d1, d2, d3, d4]
        pd = self.fcrn(x)
        loss = self.loss(pd, y)
        train_mce = mce(pd, y)
        self.log('train_loss', loss)
        self.log('train_mce', train_mce, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, d1, d2, d3, d4 = val_batch
        y = [d1, d2, d3, d4]
        pd = self.fcrn(x)
        loss = self.loss(pd, y)
        val_mce = mce(pd, y)
        self.log('val_loss', loss)
        self.log('val_mce', val_mce, on_epoch=True, prog_bar=True, logger=True)
        
    def predict_step(self, batch, batch_idx):
        x, _, _, _, _ = batch
        pred = self.fcrn(x)[-1]
        
        return pred, _


def norm_0_1(mask):
    x_min = np.min(mask)
    x_max = np.max(mask)
    
    new_mask = (mask-x_min)/(x_max-x_min)
    
    return new_mask

def get_cell_points(den_map, min_dis=2, thres=0.4):
    
    if np.max(den_map) < 0.4:
        return []
    
    den_map = norm_0_1(den_map)
    
    x_y = peak_local_max(
        den_map, 
        min_distance = min_dis,
        threshold_abs = thres,
    )
    
    return x_y
    
def save_csv(_dict):
    try:
        img = list(_dict.keys())[0]
        pred = _dict[img]
        x_y = get_cell_points(pred)
        name = img.split('/')[-1]
        df = pd.DataFrame(x_y[:, ::-1])
        df.to_csv('{}/{}.csv'.format(SAVE_PATH, name), index=None, header=None)
    except:
        print('None or Error!', list(_dict.keys())[0])
        
# main

if __name__ == '__main__':
    df_list = np.array(glob.glob('{}/*.csv'.format(DF_PATH)))
    
    for df_path in df_list:
        df = pd.read_csv(df_path)
        base_name = df_path.split('/')[-1].split('.')[0]
        
        test_dataset = MyInferDataset(df, dim=DIM, sigma=SIGMA)
        test_loader = DataLoader(
            dataset = test_dataset,
            batch_size = BS,
            shuffle = False,
            num_workers = NW,
        )

        model = FCRN(IN_CHANNELS)

        # training
        trainer = pl.Trainer(
            gpus = GPUS,
            max_epochs = EPOCHS,
            logger = False,
        )

        # inference
        predictions = trainer.predict(
            model = model,
            dataloaders = test_loader, 
            ckpt_path = ckpt_path
        )

        preds = torch.squeeze(torch.concat([item[0] for item in predictions])).numpy().tolist()
        dict_list = [{img: pred} for pred, img in zip(preds, df['img'].tolist())]
        
        with Pool(processes=NW*2) as p:
            p.map(save_csv, dict_list)
        
            

            
        