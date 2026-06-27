import streamlit as st
import requests

# Configure page layouts
st.set_page_config(page_title="Financial Intelligence Dashboard", layout="wide")

st.title("📊 Financial Intelligence Platform")
st.markdown("Dual-Model Engine: Local BERT Sentiment Classifier + LangChain RAG Processing.")

# Define Backend FastAPI URLs (Update ports if different)
BACKEND_URL = "http://localhost:8001"

col1, col2 = st.columns(2)

# --- PHASE 1: BERT SENTIMENT CLASSIFICATION ---
with col1:
    st.header("🎯 Real-Time Sentiment Analysis")
    st.subheader("Powered by Local BERT & PyTorch")
    
    sentiment_input = st.text_area(
        "Enter financial text statement:", 
        placeholder="e.g., Operating profit margin was 8.3%, compared to 11.7% in the previous cycle.",
        key="sentiment_text"
    )
    
    if st.button("Analyze Sentiment", type="primary"):
        if sentiment_input.strip():
            with st.spinner("Running local Transformer inference..."):
                try:
                    res = requests.post(f"{BACKEND_URL}/predict", json={"text": sentiment_input})
                    if res.status_code == 200:
                        prediction = res.json().get("predicted_sentiment", "Unknown")
                        st.success(f"**Predicted Sentiment:** {prediction}")
                    else:
                        st.error(f"Error: Received status code {res.status_code}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")
        else:
            st.warning("Please enter text before running analysis.")

# --- PHASE 2: LANGCHAIN RAG ENGINE ---
with col2:
    st.header("🔍 Grounded Financial Q&A")
    st.subheader("Powered by LangChain, Qdrant, & Gemini")
    
    rag_query = st.text_input(
        "Ask a specific financial query:",
        placeholder="e.g.,What is the closest recorded operating profit margin percentage? Also, list the raw context segments you used to find this answer.",
        key="rag_input"
    )
    
    if st.button("Query RAG System", type="primary"):
        if rag_query.strip():
            with st.spinner("Retrieving vector context and generating response..."):
                try:
                    res = requests.post(f"{BACKEND_URL}/rag", json={"query": rag_query})
                    if res.status_code == 200:
                        answer = res.json().get("answer", "No response payload received.")
                        st.info(f"**Engine Response:**\n\n{answer}")
                    else:
                        st.error(f"Error: Received status code {res.status_code}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")
        else:
            st.warning("Please enter a query statement.")