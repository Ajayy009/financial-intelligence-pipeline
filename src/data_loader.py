import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class FinancialDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=64):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        sentence = str(row['sentence'])
        label = int(row['label'])

        # Hugging Face tokenization optimized for PyTorch (return_tensors="pt")
        encoding = self.tokenizer(
            sentence,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def prepare_financial_dataset(model_name="bert-base-uncased", batch_size=2, train_split_ratio=0.75):
    print("📡 Loading data from local workspace file for PyTorch pipeline...")
    local_path = os.path.join("data", "financial_data.csv")
    
    df = pd.read_csv(local_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_size = int(len(df) * train_split_ratio)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    print(f"📊 Dataset Split: {len(train_df)} train samples, {len(val_df)} validation samples.")
    print(f"📥 Initializing Tokenizer from Hugging Face: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = FinancialDataset(train_df, tokenizer)
    val_dataset = FinancialDataset(val_df, tokenizer)

    # PyTorch native DataLoaders handle shuffling and batching beautifully
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print("✅ PyTorch Ingestion DataLoaders Configured Successfully.")
    return train_loader, val_loader, tokenizer

if __name__ == "__main__":
    train_loader, val_loader, tok = prepare_financial_dataset(batch_size=2)
    
    # Quick Smoke Test to inspect a PyTorch batch matrix
    for batch in train_loader:
        print("\n--- PyTorch Tensor Structure Inspection ---")
        print(f"Input IDs Shape:      {batch['input_ids'].shape}")
        print(f"Attention Mask Shape: {batch['attention_mask'].shape}")
        print(f"Labels Shape:         {batch['label'].shape}")
        break