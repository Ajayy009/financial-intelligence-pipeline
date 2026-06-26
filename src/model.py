import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class FinancialTransformerClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_classes=3):
        super(FinancialTransformerClassifier, self).__init__()
        print(f"育️ Loading native PyTorch Hugging Face Transformer backbone: {model_name}...")
        
        # This pulls down the pre-trained BERT weights and sets up a classification head automatically
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_classes
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # Returns raw output logits for classification tracking
        return outputs.logits

if __name__ == "__main__":
    # Local Smoke Test to verify structural layout and sizing
    model = FinancialTransformerClassifier()
    print("✅ PyTorch Hugging Face Model Initialized Successfully.")
    print(model)
    