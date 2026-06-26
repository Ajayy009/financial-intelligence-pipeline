import os
import torch
import torch.nn as nn
from torch.optim import AdamW
import mlflow  # <-- 1. Import MLflow
from data_loader import prepare_financial_dataset
from model import FinancialTransformerClassifier

def train_pipeline(epochs=1, batch_size=2, learning_rate=2e-5):
    print("🚀 Initializing Phase 2: Training Pipeline Setup with MLflow...")
    
    # Set the experiment name in MLflow
    mlflow.set_experiment("Financial_Sentiment_BERT")
    
    # Start tracking this specific training run
    with mlflow.start_run():
        # 2. Log training parameters/hyperparameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        
        # Load Data Streams
        train_loader, val_loader, _ = prepare_financial_dataset(batch_size=batch_size)
        
        # Instantiate Model Topology
        model = FinancialTransformerClassifier(num_classes=3)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"💻 Running training routine on hardware target: {device}")
        
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Master Epoch Loop Iteration
        for epoch in range(epochs):
            print(f"\n🎬 --- Epoch {epoch + 1} of {epochs} ---")
            
            # --- TRAINING PHASE ---
            model.train()
            total_train_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                loss = criterion(logits, labels)
                total_train_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                
            avg_train_loss = total_train_loss / len(train_loader)
            print(f"📉 Train Loss Summary: {avg_train_loss:.4f}")
            
            # --- VALIDATION PHASE ---
            model.eval()
            total_val_loss = 0
            correct_predictions = 0
            total_samples = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    token_type_ids = batch['token_type_ids'].to(device)
                    labels = batch['label'].to(device)
                    
                    logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                    loss = criterion(logits, labels)
                    total_val_loss += loss.item()
                    
                    predictions = torch.argmax(logits, dim=1)
                    correct_predictions += torch.sum(predictions == labels).item()
                    total_samples += labels.size(0)
                    
            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = (correct_predictions / total_samples) * 100
            print(f"📊 Val Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.2f}%")
            
            # 3. Log real-time metrics to MLflow dashboard at the end of the epoch
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

    print("\n🏆 Local MLflow Smoke Test Execution Finished Successfully!")

    # 💾 Auto-Save Block: Serializes model weights to your local workspace
    output_dir = "./saved_model"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the underlying PyTorch weights from your model architecture
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    print(f"✅ Production weights successfully saved locally to: {output_dir}/pytorch_model.bin")

if __name__ == "__main__":
    # Update to your real training configurations
    train_pipeline(epochs=2, batch_size=32)