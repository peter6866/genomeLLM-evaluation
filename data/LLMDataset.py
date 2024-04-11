import torch
from torch.utils.data import Dataset
import pandas as pd


class LLMDataset(Dataset):
    def __init__(self, data_type, tokenizer=None):
        """
        Initialize the Dataset.

        Parameters:
        - data_path: path to the activity data.
        - retinopathy_path: path to the retinopathy data.
        - data_type: type of data ("train", "validate", "test", etc.).
        - tokenizer: tokenizer instance for tokenization.
        """
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided.")
        
        self.tokenizer = tokenizer
        self.sequence_key = "Sequence"
        activity_key = "Dev_log2_enrichment"
        
        if data_type == "train":
            data_df = pd.read_csv('data/train_data.csv')
        elif data_type == "val":
            data_df = pd.read_csv('data/val_data.csv')
        elif data_type == "test":
            data_df = pd.read_csv('data/test_data.csv')
     

        self.sequence = data_df[self.sequence_key]
        self.target = torch.tensor(data_df[activity_key].values, dtype=torch.float)

        # Tokenize the sequences during initialization
        self.encodings = self.tokenizer.batch_encode_plus(
            list(self.sequence),
            add_special_tokens=True,
            max_length=35,
            truncation=True,
            return_token_type_ids=False,
            padding=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
    def __len__(self):
        return len(self.sequence)
    
    def __getitem__(self, idx):
        # Use the pre-tokenized sequences and attention masks
        input_ids = self.encodings['input_ids'][idx]
        attention_mask = self.encodings['attention_mask'][idx]

        res = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': self.target[idx]
        }

        return res