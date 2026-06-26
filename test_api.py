import pytest
from fastapi.testclient import TestClient

# Import your FastAPI app instance directly from our src folder
from src.main import app

client = TestClient(app)

def test_health_check_endpoint():
    """Verify the root endpoint is running and reporting healthy status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_sentiment_classification_endpoint():
    """Verify BERT model outputs a valid dictionary payload."""
    payload = {"text": "The company reported excellent third-quarter revenue growth."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_sentiment" in data
    assert data["text"] == payload["text"]