import pytest
from unittest.mock import patch, MagicMock

# 1. Mock the heavy local startup tasks so the app boots instantly on GitHub
with patch("transformers.AutoTokenizer.from_pretrained"), \
     patch("torch.load"), \
     patch("src.model.FinancialTransformerClassifier"), \
     patch("src.rag.rag_engine.FinancialRAGEngine"):
     
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
    """Verify BERT model outputs a valid dictionary payload using a mock."""
    with patch("src.main.predict_sentiment") as mock_predict:
        mock_predict.return_value = "Positive"
        
        payload = {"text": "The company reported excellent third-quarter revenue growth."}
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "predicted_sentiment" in data
        assert data["text"] == payload["text"]

def test_rag_query_endpoint():
    """Verify the LangChain RAG engine route processes queries and handles responses."""
    # We mock the global rag_engine instance inside src.main
    with patch("src.main.rag_engine") as mock_rag:
        # Simulate what your verified answer_query() method returns
        mock_rag.answer_query.return_value = "Mocked Response: Operating profit margins grew by 11.7%."
        
        payload = {"query": "What was the operating profit margin?"}
        response = client.post("/rag", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "query" in data
        assert data["query"] == payload["query"]