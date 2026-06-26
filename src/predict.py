import os
import torch
from transformers import AutoTokenizer
from src.model import FinancialTransformerClassifier

def predict_sentiment(text, model, tokenizer, device):
    # 1. Tokenize the input text exactly how the model expects it
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True
    )
    
    # Move inputs to the correct hardware target
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    token_type_ids = inputs['token_type_ids'].to(device)
    
    # 2. Run forward pass (Inference)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        prediction = torch.argmax(logits, dim=1).item()
    
    # 3. Map numerical class back to string label
    # (Assuming: 0 = Negative, 1 = Neutral, 2 = Positive)
    label_mapping = {0: "🔴 Negative", 1: "🟡 Neutral", 2: "🟢 Positive"}
    return label_mapping.get(prediction, "Unknown")

if __name__ == "__main__":
    print("🔮 Loading your locally saved 84.30% accuracy model...")
    
    # Define paths pointing to your saved local assets
    model_dir = "./saved_model"
    weights_path = os.path.join(model_dir, "pytorch_model.bin")
    
    # Set up hardware target
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the optimized tokenizer from your local folder
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Instantiate the model architecture and load the weights
    model = FinancialTransformerClassifier(num_classes=3)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    print("✅ Model and Tokenizer loaded successfully offline!\n")
    
    # Test Sentences
    test_headlines = [
        "Tech Mahindra shares surged by 5% following a spectacular quarterly profit report.",
        "The central bank raised interest rates unexpectedly, sparking fears of a market slowdown.",
        "The company announced it will maintain its current production levels for the upcoming fiscal year."
    ]
    
    print("🚀 Running Local Live Inference Tests:")
    print("-" * 60)
    for headline in test_headlines:
        sentiment = predict_sentiment(headline, model, tokenizer, device)
        print(f"📝 Text: {headline}")
        print(f"🧠 Predicted Sentiment: {sentiment}\n")