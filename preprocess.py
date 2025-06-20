import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Define dataset class
class ComplaintDataset(Dataset):
    def __init__(self, complaints, labels, tokenizer, max_length):
        self.complaints = complaints
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.complaints)

    def __getitem__(self, idx):
        complaint = self.complaints[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            complaint,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# Function to preprocess data
def preprocess_data(file_path, max_length=128, batch_size=16):
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Load data
    df = pd.read_csv(file_path)
    
    # Encode labels
    label_encoder = LabelEncoder()
    df["category_encoded"] = label_encoder.fit_transform(df["category"])
    
    # Save label encoder
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Save tokenizer for later use
    tokenizer.save_pretrained("models/bert_tokenizer")

    # Dataset & DataLoader
    dataset = ComplaintDataset(df["complaint"].tolist(), df["category_encoded"].tolist(), tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, label_encoder  # Now returning label_encoder

if __name__ == "__main__":
    preprocess_data("customer_complaints.csv")
