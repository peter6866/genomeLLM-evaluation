import time
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import wandb

from data.DataModule import DeepSTARRDataModule
from model.DeepSTARRModel import DeepSTARR_Lightning

def train(seed):
    # Global parameters
    params_smaller_data = {'batch_size': 64, # number of examples per batch
                        'epochs': 100, # number of epochs
                        'early_stop': 10, # patience of 10 epochs to reduce training time; you can increase the patience to see if the model improves after more epochs
                        'lr': 0.001, # learning rate
                        'n_conv_layer': 3, # number of convolutional layers
                        'num_filters1': 128, # number of filters/kernels in the first conv layer
                        'num_filters2': 60, # number of filters/kernels in the second conv layer
                        'num_filters3': 60, # number of filters/kernels in the third conv layer
                        'kernel_size1': 7, # size of the filters in the first conv layer
                        'kernel_size2': 3, # size of the filters in the second conv layer
                        'kernel_size3': 5, # size of the filters in the third conv layer
                        'n_dense_layer': 1, # number of dense/fully connected layers
                        'dense_neurons1': 64, # number of neurons in the dense layer
                        'dropout_conv': 'yes', # add dropout after convolutional layers?
                        'dropout_prob': 0.4, # dropout probability
                        'pad':'same'}


    pl.seed_everything(seed)

    wandb.login()

    wandb_logger = WandbLogger(
        project='561s-final-project',
        name=time.strftime('%Y-%m-%d-%H-%M') + "-CNN",
        )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00,
        patience=params_smaller_data['early_stop'], 
        verbose=False,
        mode="min")

    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator='gpu',
        devices=-1,
        max_epochs=params_smaller_data['epochs'],
        deterministic=True,
        fast_dev_run=False,
        callbacks=[early_stop_callback]
        )

    data_module = DeepSTARRDataModule(
        batch_size=params_smaller_data['batch_size']
        )

    model = DeepSTARR_Lightning(
        params_smaller_data
    )

    torch.set_float32_matmul_precision('high')

    trainer.fit(model, data_module)

    trainer.test(model, data_module)
    test_pcc = trainer.callback_metrics.get('test_pcc')
   
    wandb.finish()

    return test_pcc.item()

if __name__ == '__main__':
    test_pccs = []
    for seed in range(1, 6):
        test_pcc = train(seed)
        test_pccs.append(test_pcc)
    
    print(test_pccs)