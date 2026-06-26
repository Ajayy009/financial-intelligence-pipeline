import pytest
from unittest.mock import patch, MagicMock

# 1. We mock the heavy startup tasks BEFORE importing the app instance
# This prevents the app from crashing due to missing 'saved_model' files on GitHub
with patch("transformers.AutoTokenizer.from_pretrained"), \
     patch("torch.load"), \
     patch("src.model.FinancialTransformerClassifier"), \
     patch("src.rag.rag_engine.FinancialRAGEngine"):
     
    # Now we safely import the app without it throwing a startup error
    from src.main import app

from fastapi.testclient import TestClient
client = TestClient(app)

def test_health_check_endpoint():
    """Verify the root endpoint is running and reporting healthy status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_sentiment_classification_endpoint():
    """Verify the endpoint returns a valid dictionary payload using a mock."""
    # Mock the internal predict_sentiment function execution
    with patch("src.main.predict_sentiment") as mock_predict:
        mock_predict.return_value = "Positive"
        
        payload = {"text": "The company reported excellent third-quarter revenue growth."}
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "predicted_sentiment" in data
        assert data["text"] == payload["text"]