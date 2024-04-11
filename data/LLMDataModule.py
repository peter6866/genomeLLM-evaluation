import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .LLMDataset import LLMDataset


class LLMDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size):
        super(LLMDataModule, self).__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
     
        # Create datasets
        self.train_dataset = LLMDataset("train", self.tokenizer)
        self.validate_dataset = LLMDataset("val", self.tokenizer)
        self.test_dataset = LLMDataset("test", self.tokenizer)
        

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        
        return DataLoader(self.validate_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False, drop_last=True, pin_memory=True)
       
    def test_dataloader(self):
       
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False, pin_memory=True)
        
        