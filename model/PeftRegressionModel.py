import pytorch_lightning as pl
import torchmetrics
import torch
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import numpy as np


class PeftRegressionModel(pl.LightningModule):
    def __init__(self, model_name, train_dataloader_len, num_epochs, lr=3e-3):
        super(PeftRegressionModel, self).__init__()

        # Model
        self.model = model_name
        print(self.model.print_trainable_parameters())

        # Hyperparameters
        self.lr = lr
        self.train_dataloader_len = train_dataloader_len
        self.num_epochs = num_epochs

        # Regression Metrics
        self.train_pcc = torchmetrics.PearsonCorrCoef()
        self.val_pcc = torchmetrics.PearsonCorrCoef()
        self.test_pcc = torchmetrics.PearsonCorrCoef()
       
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = self.loss_fn(outputs.logits.squeeze(), batch["labels"])
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.train_pcc.update(outputs.logits.squeeze(), batch["labels"])
        return loss
    
    def on_train_epoch_end(self):
        self.log('train_pcc', self.train_pcc.compute())
        self.train_pcc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0.06 * (self.train_dataloader_len * self.num_epochs),
            num_training_steps=(self.train_dataloader_len * self.num_epochs)
        )
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = self.loss_fn(outputs.logits.squeeze(), batch["labels"])
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        self.val_pcc.update(outputs.logits.squeeze(), batch["labels"])
    
    def on_validation_epoch_end(self):
        self.log('val_pcc', self.val_pcc.compute())
        self.val_pcc.reset()

    def test_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        pred = outputs.logits.squeeze()
      
        self.test_pcc.update(pred, batch["labels"])

    def on_test_epoch_end(self):
        self.log('test_pcc', self.test_pcc.compute())
        self.test_pcc.reset()
  