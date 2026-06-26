import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 🌟 1. Import MLflow Tracking API
import mlflow

# LangChain Orchestration Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = 6333
COLLECTION_NAME = "financial_phrases"
EMBEDDING_MODEL_NAME = "./saved_model/all-MiniLM-L6-v2"

class FinancialRAGEngine:
    def __init__(self):
        # 🌟 2. Set up the tracking database server connection cleanly
        mlflow.set_tracking_uri("sqlite:///mlflow.db?timeout=20")
        mlflow.set_experiment("Financial_Intelligence_Pipeline")
        
        # 🌟 3. Enable absolute automatic telemetry for all LangChain actions
        mlflow.langchain.autolog()
        
        # 1. Connect to running Qdrant DB
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # 2. Load local embedding model
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # 3. Initialize LangChain's Gemini Wrapper (picks up GEMINI_API_KEY from env)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    def retrieve_context(self, query: str, limit: int = 3):
        """Converts query to vector and searches Qdrant using query_points API."""
        query_vector = self.embedder.encode(query).tolist()
        
        search_results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit
        )
        
        contexts = []
        for hit in search_results.points:
            if hit.payload:
                text = hit.payload.get("text_content") or hit.payload.get("sentence")
                if text:
                    contexts.append(text)
        return contexts

    def answer_query(self, query: str) -> str:
        """Orchestrates Context Retrieval + Gemini Generation using LangChain LCEL."""
        # Step A: Fetch your verified DB matches
        contexts = self.retrieve_context(query, limit=3)
        context_str = "\n".join([f"- {c}" for c in contexts])
        
        # Step B: Define LangChain Prompt Template
        prompt = ChatPromptTemplate.from_template("""
You are an expert Financial Analyst AI. Answer the user's question using ONLY the provided verified context segments.
If the context doesn't contain a direct answer, summarize what the closest facts state. Be direct and concise.

Context Data:
{context}

User Question: {question}
Answer:
""")
        
        # Step C: Chain components together using LangChain Expression Language (LCEL)
        chain = prompt | self.llm | StrOutputParser()
        
        # Step D: Run the framework execution
        response = chain.invoke({"context": context_str, "question": query})
        return response

# Interactive Terminal Test Runner
if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        print("⚠️ Warning: GEMINI_API_KEY environment variable not found!")
        
    print("🤖 Initializing LangChain RAG Engine...")
    engine = FinancialRAGEngine()
    print("✅ Engine Ready!")
    print("-" * 50)
    print("Type your financial query below. Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user_query = input("🔍 Enter Financial Query: ").strip()
            
            # Check for exit command
            if user_query.lower() in ['exit', 'quit']:
                print("\n👋 Exiting RAG Engine. Happy analyzing!")
                break
                
            if not user_query:
                continue
                
            print("\n⚙️ Processing retrieval and generation...")
            answer = engine.answer_query(user_query)
            
            print("\n🤖 Gemini RAG Answer:")
            print(answer)
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Exiting RAG Engine.")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            print("-" * 50)