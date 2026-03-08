import torch, os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils import read_jsonl
from typing import List

class MemoryDataset(Dataset):
    def __init__(self, 
                 json_file: str, 
                 max_length:int, 
                 tokenizer: AutoTokenizer, 
                 root_dir=None):
        """
        Args:
            json_file (str): Path to the JSON file containing the data.
        """
        self.data: List = read_jsonl(json_file)
        self.max_length: int = max_length
        self.tokenizer: AutoTokenizer = tokenizer


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        content, label = self.data[idx]['text'], self.data[idx]['label']
        encoding = self.tokenizer(
            content,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
