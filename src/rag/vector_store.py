import os
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# 1. Configuration Constants
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "financial_phrases"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CSV_PATH = "./data/financial_data.csv"

def initialize_vector_store():
    """
    Connects to Qdrant, reads financial_data.csv, chunks text, 
    generates embeddings locally, and upserts them into a clean collection.
    """
    print("🚀 Initializing Vector Store Build Pipeline...")
    
    # Check if data exists
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"❌ Missing data source file at: {CSV_PATH}")
        
    # Connect to local Qdrant container
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Initialize the local embedding transformer (runs lightweight on CPU)
    print(f"📦 Loading local embedding engine ({EMBEDDING_MODEL_NAME})...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Setup standard vector configurations (384 dimensions for all-MiniLM-L6-v2)
    vector_config = VectorParams(size=384, distance=Distance.COSINE)
    
    # Recreate collection cleanly
    print(f"🔄 Re-creating collection '{COLLECTION_NAME}' in Qdrant VDB...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=vector_config
    )
    
    # Load raw text dataset
    print(f"📖 Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Assuming standard column naming or fall back to first text column
    text_column = "sentence" if "sentence" in df.columns else df.columns[0]
    raw_sentences = df[text_column].dropna().astype(str).tolist()
    
    print(f"✂️ Chunking {len(raw_sentences)} sentences using LangChain processing...")
    # Clean chunk split architecture
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    points = []
    point_id = 1
    
    for idx, sentence in enumerate(raw_sentences):
        # Split text chunk dynamically
        chunks = text_splitter.split_text(sentence)
        
        for chunk in chunks:
            # Generate vectors
            vector = embedder.encode(chunk).tolist()
            
            # Format payload payload structure for Qdrant storage
            payload = {
                "original_index": idx,
                "text_content": chunk
            }
            
            # Pack structurally as a Qdrant storage point
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            )
            point_id += 1

    # Bulk upload payload points into Qdrant 
    print(f"📤 Upserting {len(points)} vector structures into Qdrant Core...")
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i:i + batch_size]
        )
        
    print(f"✅ Vector database generation complete! Cleanly injected into '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    initialize_vector_store()