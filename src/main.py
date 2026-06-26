import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <-- 1. Added CORS Import
from pydantic import BaseModel
from transformers import AutoTokenizer
from src.model import FinancialTransformerClassifier
from src.predict import predict_sentiment

# Phase 2 Imports: Import our RAG Engine class
from src.rag.rag_engine import FinancialRAGEngine

# 1. Initialize FastAPI App
app = FastAPI(
    title="Financial Intelligence NLP & RAG API",
    description="Production-ready API providing BERT sentiment classification and GenAI RAG capabilities.",
    version="2.0.0"
)

# <-- 2. Added CORS Middleware Configuration so Swagger works perfectly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your local browser to communicate with the container
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST)
    allow_headers=["*"],
)

# 2. Define the Request Payload Structures using Pydantic
class SentimentRequest(BaseModel):
    text: str

class RAGRequest(BaseModel):
    query: str

# 3. Global variables to hold our model and engine components
model = None
tokenizer = None
device = None
rag_engine = None  # Global placeholder for Phase 2 Engine

@app.on_event("startup")
def load_model_assets():
    """
    This block runs automatically when the FastAPI server starts up.
    It loads BOTH the BERT weights and the LangChain RAG engine into memory once.
    """
    global model, tokenizer, device, rag_engine
    print("🔮 API Server Starting: Loading multi-model assets into memory...")
    
    # --- PHASE 1: Load BERT Assets ---
    model_dir = "./saved_model"
    weights_path = os.path.join(model_dir, "pytorch_model.bin")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = FinancialTransformerClassifier(num_classes=3)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()
        print("✅ Phase 1: BERT sentiment assets safely loaded.")
    except Exception as e:
        print(f"❌ Critical Error loading BERT assets: {str(e)}")
        raise RuntimeError(e)

    # --- PHASE 2: Load LangChain RAG Engine ---
    try:
        print("🤖 Phase 2: Initializing LangChain RAG Engine...")
        # Instantiating the engine runs the __init__ logic (setting up Qdrant client & Gemini)
        rag_engine = FinancialRAGEngine()
        print("✅ Phase 2: RAG Engine successfully initialized.")
    except Exception as e:
        print(f"❌ Critical Error initializing RAG Engine: {str(e)}")
        raise RuntimeError(e)

    print("🚀 All systems online! API is ready to accept production requests.")

# 4. Define a Root Health Check Endpoint
@app.get("/")
def read_root():
    return {
        "status": "healthy", 
        "models_loaded": {
            "classification": "Financial_Sentiment_BERT_84.30%",
            "rag_llm": "gemini-2.5-flash"
        }
    }

# 5. Define Phase 1: Core Prediction Endpoint
@app.post("/predict")
def get_prediction(payload: SentimentRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
    
    try:
        sentiment = predict_sentiment(payload.text, model, tokenizer, device)
        return {
            "text": payload.text,
            "predicted_sentiment": sentiment
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")

# 6. Define Phase 2: Core RAG Query Endpoint
@app.post("/rag")
def get_rag_answer(payload: RAGRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG Engine is uninitialized or unavailable.")
    
    try:
        # Calling your verified method name here
        answer = rag_engine.answer_query(payload.query)
        return {
            "query": payload.query,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Pipeline Error: {str(e)}")