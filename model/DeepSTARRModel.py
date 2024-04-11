import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.optim import Adam
import torchmetrics

class DeepSTARR_Lightning(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)

        # Convolutional layers setup
        layers = []

        # Initial Conv layer
        layers.append(nn.Conv1d(in_channels=4, out_channels=params['num_filters1'],
                                kernel_size=params['kernel_size1'], padding=params['pad']))
        layers.append(nn.BatchNorm1d(params['num_filters1']))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(kernel_size=2))

        # Additional Conv layers
        for i in range(1, params['n_conv_layer']):
            layers.append(nn.Conv1d(in_channels=params['num_filters'+str(i)],
                                    out_channels=params['num_filters'+str(i+1)],
                                    kernel_size=params['kernel_size'+str(i+1)],
                                    padding=params['pad']))
            layers.append(nn.BatchNorm1d(params['num_filters'+str(i+1)]))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            if params['dropout_conv'] == 'yes':
                layers.append(nn.Dropout(params['dropout_prob']))

        self.conv_layers = nn.Sequential(*layers)

        # Fully connected layers setup
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1860, params['dense_neurons1']),  
            nn.BatchNorm1d(params['dense_neurons1']),
            nn.ReLU(),
            nn.Dropout(params['dropout_prob']),
            nn.Linear(params['dense_neurons1'], 1)  # Single task output
        )

        # Metrics
        self.train_pcc = torchmetrics.PearsonCorrCoef()
        self.val_pcc = torchmetrics.PearsonCorrCoef()
        self.test_pcc = torchmetrics.PearsonCorrCoef()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.train_pcc.update(y_hat, y)
        return loss

    def on_train_epoch_end(self):
        self.log('train_pcc', self.train_pcc.compute())
        self.train_pcc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.val_pcc.update(y_hat, y)
        
    def on_validation_epoch_end(self):
        self.log('val_pcc', self.val_pcc.compute())
        self.val_pcc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        self.test_pcc.update(y_hat, y)

    def on_test_epoch_end(self,):
        self.log('test_pcc', self.test_pcc.compute())
        self.test_pcc.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams['lr'])
