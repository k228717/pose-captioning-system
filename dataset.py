import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer

class PoseDataset(Dataset):
    def __init__(self, csv_path, tokenizer_name="t5-small", max_len=64):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

        # adjust if your columns different
        self.captions = self.data.iloc[:, 2]
        self.pose_data = self.data.iloc[:, 3:].values.astype("float32")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        caption = str(self.captions[idx])
        pose = self.pose_data[idx]

        pose_tensor = torch.tensor(pose)

        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "pose": pose_tensor,
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze()
        }