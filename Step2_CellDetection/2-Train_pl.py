import os
import json
import glob
import torch
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
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

# constants

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
IMAGE_PATH = '**/Step2_CellDetection/dataset/data'


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
        x, _, _, _, d4 = batch
        pred = self.fcrn(x)[-1]
        
        return pred, d4


if __name__ == '__main__':
    train_img_list = np.array(glob.glob('{}/{}/*.jpg'.format(IMAGE_PATH, 'train'))).tolist()
    eval_img_list = np.array(glob.glob('{}/{}/*.jpg'.format(IMAGE_PATH, 'eval'))).tolist()
    test_img_list = np.array(glob.glob('{}/{}/*.jpg'.format(IMAGE_PATH, 'test'))).tolist()

    train_dataset = MyDataset(train_img_list, dim=DIM, sigma=SIGMA, data_type='train')
    eval_dataset = MyDataset(train_img_list, dim=DIM, sigma=SIGMA, data_type='eval')
    test_dataset = MyDataset(test_img_list, dim=DIM, sigma=SIGMA, data_type='test')

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = BS,
        shuffle = True,
        num_workers = NW,
        drop_last = True,
    )

    eval_loader = DataLoader(
        dataset = eval_dataset,
        batch_size = BS,
        shuffle = False,
        num_workers = NW,
    )
    
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = BS,
        shuffle = False,
        num_workers = NW,
    )

    model = FCRN(IN_CHANNELS)
    
    logger = TensorBoardLogger(
        name = DATASETS,
        save_dir = 'lightning_logs',
    )

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs = EVERY_N_EPOCHS,
        save_top_k = SAVE_TOP_K,
        monitor = 'val_mce',
        mode = 'min',
        save_last = True,
        filename = '{epoch}-{val_mce:.2f}'
    )

    earlystop_callback = EarlyStopping(
        monitor = "val_mce", 
        mode = "min",
        min_delta = 0.00, 
        patience = EARLY_STOP,
    )

    # training
    trainer = pl.Trainer(
        gpus = GPUS,
        max_epochs = EPOCHS,
        logger = logger,
        log_every_n_steps = LOG_EVERY_N_STEPS,
        callbacks = [checkpoint_callback, earlystop_callback],
    )

    trainer.fit(
        model, 
        train_loader, 
        eval_loader
    )

    # inference
    predictions = trainer.predict(
        dataloaders = test_loader, 
        ckpt_path = 'best'
    )

    preds = torch.squeeze(torch.concat([item[0] for item in predictions])).numpy().tolist()
    labels = torch.squeeze(torch.concat([item[1] for item in predictions])).numpy().tolist()

    results ={
        'img_path': test_img_list,
        'pred': preds,
        'label': labels,
    }

    results_json = json.dumps(results)
    with open(os.path.join(trainer.log_dir, 'result.json'), 'w+') as f:
        f.write(results_json)
