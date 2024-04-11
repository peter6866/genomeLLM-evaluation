import time
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import wandb
import peft
from peft import IA3Config, LoraConfig

from data.LLMDataModule import LLMDataModule

from model.PeftRegressionModel import PeftRegressionModel


BATCH_SIZE = 32
MAX_EPOCHS = 10
LEARNING_RATE = 5e-4

# Import the tokenizer and the model
pretrained_model_name = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
# TODO: FIXIT
# pretrained_model_name = "zhihan1996/DNABERT-2-117M"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
model_O = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name, 
        num_labels=1, 
        hidden_dropout_prob=0.4
    )

# IA3 config
# peft_config = IA3Config(
#     task_type="SEQ_CLS",
#     inference_mode=False,
#     target_modules=['key', 'value', 'dense'],
#     feedforward_modules=['dense'])
# model_O = peft.get_peft_model(model_O, peft_config)

# LORA config
lora_config = LoraConfig(
    task_type="SEQ_CLS",
    lora_alpha=32,
    lora_dropout=0.1,
    r=16,
    bias="none",
    # target_modules=["query", "value", "dense"],
    target_modules=["query", "dense"],
    inference_mode=False,
)
model_O = peft.get_peft_model(model_O, lora_config)

pl.seed_everything(1)

wandb.login()

wandb_logger = WandbLogger(
    project='561s-final-project',
    name=time.strftime('%Y-%m-%d-%H-%M') + "-LLM",
    )

early_stop_callback = EarlyStopping(
    monitor="val_loss", 
    min_delta=0.00,
    patience=5, 
    verbose=False,
    mode="min"
)

trainer = pl.Trainer(
    logger=wandb_logger,
    accelerator='gpu',
    devices=-1,
    max_epochs=MAX_EPOCHS,
    deterministic=True,
    fast_dev_run=False,
    callbacks=[early_stop_callback],
    precision=16
    )

data_module = LLMDataModule(
     tokenizer=tokenizer,
     batch_size=BATCH_SIZE)

model = PeftRegressionModel(model_O, len(data_module.train_dataloader()), MAX_EPOCHS, lr=LEARNING_RATE)

torch.set_float32_matmul_precision('high')
trainer.fit(model, data_module)
trainer.test(model, data_module)

wandb.finish()