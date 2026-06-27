
# Dual-Engine Financial Intelligence Platform

An end-to-end, production-grade **MLOps & LLMOps** platform featuring an integrated AI architecture. The system unifies a fine-tuned local Transformer model for real-time risk classification with a Generative AI Retrieval-Augmented Generation (RAG) pipeline, backed by automated **CI/CD** infrastructure for seamless verification and deployment.

---

## 🏗️ Architecture Overview

The platform integrates two high-performance operational phases exposed via a single, unified containerized FastAPI backend and a decoupled Streamlit frontend dashboard:

### Phase 1: Real-Time Financial Sentiment Engine
* **Core Model:** Fine-tuned `bert-base-uncased` via Hugging Face and PyTorch.
* **Training & Compute:** Trained using a **Google Colab T4 GPU** environment. The optimized model weights (`model.pt`) were exported and packed into our local production inference layer.
* **Objective:** Analyzes volatile financial text statements and provides instantaneous sentiment classification (`Positive`, `Negative`, `Neutral`) locally, ensuring ultra-low latency without external network dependencies.

#### 📊 Model Training Metrics (2 Epochs)
| Epoch | Training Loss | Validation Loss | Accuracy |
| :--- | :--- | :--- | :--- |
| **Epoch 1** | 0.6917 |  0.4621  | 82.15% |
| **Epoch 2** | 0.3462 | 0.4151 | 84.30%| 

### Phase 2: Grounded Financial Q&A (GenAI RAG Engine)
* **Orchestration:** LangChain Framework.
* **Vector Storage:** Qdrant Vector Database (isolated container for high-speed dense vector similarity searches).
* **LLM Engine:** Google Gemini Pro (`gemini-1.5-flash`) via `langchain-google-genai`.
* **Objective:** Prevents hallucinations by forcing the LLM to synthesize answers grounded strictly inside extracted financial document contexts retrieved via vector similarity search.

### 🖥️ Unified Front-End Interface
Both engines are accessible side-by-side inside a responsive, real-time analytics dashboard.

*[INSERT SCREENSHOT: Full view of your Streamlit app running both the BERT sentiment classifier output and the Grounded RAG system answers simultaneously]*

---

## 🛠️ Enterprise MLOps Features & Infrastructure

### 1. Multi-Container Orchestration (Docker)
The entire ecosystem is containerized and orchestrated using **Docker Compose**. This isolates the FastAPI core engine, the Qdrant database, and the MLflow instance into a shared local network.
* FastAPI routes local traffic and hosts both AI engines seamlessly on port **8001**.

*[INSERT SCREENSHOT: Docker Desktop or terminal window showing your active, running container stack]*

### 2. Interactive Swagger API Gateway
The FastAPI application provides a single entry point exposing endpoints for both real-time model inference (`/predict`) and conversational retrieval (`/rag`).
## 📊 Platform Visuals & Interface Verification

### 1. Core Endpoints Overview (FastAPI Swagger UI)
*[INSERT SCREENSHOT: One clean shot of the overall /docs page showing both the /predict and /rag routes available]*

---

### 2. Phase 1 Engine: BERT Sentiment Analyzer (`/predict`)

| API Input Payload (Text String) | Backend API JSON Response |
| :---: | :---: |
| ![BERT Input](assets/bert_input.png) | ![BERT Response](assets/bert_response.png) |

---

### 3. Phase 2 Engine: LangChain GenAI (`/rag`)

| API Input Payload (Financial Query) | Grounded Context JSON Response |
| :---: | :---: |
| ![RAG Input](assets/rag_input.png) | ![RAG Response](assets/rag_response.png) |

---

### 4. Automated CI/CD Pipeline (GitHub Actions)
Structured via **GitHub Actions**. Every code push triggers automated dependency builds, linting validation, and an automated integration test suite (`test_api.py`) to guarantee 100% uptime before deployment.

*[INSERT SCREENSHOT: GitHub Actions tab showing your successful workflow run with the beautiful green checkmark]*

---

## 5. 📈 Experiment Tracking & Observability (MLflow)

To maintain true production-tier reliability, we integrated an active tracking layer using **MLflow**. The system tracks both phases simultaneously:
* **Phase 1 Tracking:** Monitors individual BERT classification runs, performance latencies, and output confidences.
* **Phase 2 Tracking:** Uses advanced MLflow Traces to map API endpoints, documenting execution graphs for vector lookups and LLM completions.

*[INSERT SCREENSHOT: MLflow Dashboard showing your tracking metrics, parameter configurations, or backend traces]*

---

## 📁 Repository Structure

```text
FINANCIAL_NLP_PIPELINE/
├── .github/workflows/   # CI/CD GitHub Actions pipelines
├── src/
│   ├── rag/             # LangChain & Qdrant vector store logic
│   ├── main.py          # FastAPI application & API endpoints
│   ├── model.py         # PyTorch BERT network definition
│   ├── predict.py       # BERT inference pipeline logic
│   └── train.py         # Transformer model training script
├── app.py               # Streamlit Frontend User Dashboard
├── test_api.py          # Automated CI/CD backend integration tests
├── Dockerfile           # Backend engine image blueprint
├── docker-compose.yml   # Multi-container orchestration specification
├── requirements.txt     # Python project dependencies
└── README.md            # System documentation

```

---

## 🚀 Quick Start Guide

### 1. Environment Configuration

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here

```

### 2. Launch Infrastructure Stack

Start your backend services via Docker Compose:

```bash
docker compose up -d

```

### 3. Launch UI Layer

Activate your local virtual environment and launch the presentation server:

```bash
streamlit run app.py

```

```

---

