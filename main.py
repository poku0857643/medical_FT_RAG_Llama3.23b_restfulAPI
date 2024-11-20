# Import necessary libraries
import os
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from rag_inference import RAGRetriever  # Import your RAG Retriever class
import asyncio
import logging

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Configuration variables
TOKENIZERS_PARALLELISM = os.getenv("TOKENIZERS_PARALLELISM", "false")
os.environ["TOKENIZERS_PARALLELISM"] = TOKENIZERS_PARALLELISM


# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development. Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the FT-RAG retriever
rag_retriever = RAGRetriever()

# Static file directory
static_dir = Path(__file__).parent / "static"
if not static_dir.exists():
    raise RuntimeError(f"Static directory not found: {static_dir}")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Loading RAG data...:)")
        rag_retriever.load_rag_data("./rag_file")
        logger.info("Creating embeddings and building FAISS index...")
        rag_retriever.create_embeddings()
        logger.info("FT-RAG Retriever is ready.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise SystemExit(f"Startup failed: {e}")
# Pydantic Model for query input
class QueryInput(BaseModel):
    query: str
    top_k: int = 5  # Number of documents to retrieve

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "FT-RAG service is operational"}


# FT-RAG Query Endpoint
@app.post("/ft_rag/query")
async def process_query(input_data: QueryInput):
    """
    Process a query using the FT-RAG model:
    - Retrieve top-k relevant documents using RAG
    - Generate a response using the fine-tuned model
    """
    try:
        # Retrieve context and generate response
        query = input_data.query
        top_k = input_data.top_k
        response = rag_retriever.generate_response(query=query, k=top_k)

        return {"query": query, "response": response}
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# Serve static files for the UI
static_dir = Path(__file__).parent / "static"
if not static_dir.exists():
    raise RuntimeError(f"Static directory not found: {static_dir}")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Redirect root to the static index page
@app.get("/")
async def main():
    return RedirectResponse(url="/static/index.html")


# WebSocket endpoint for real-time interaction
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            user_query = await websocket.receive_text()
            response = rag_retriever.generate_response(query=user_query)
            words = response.split()
            for word in words:
                await websocket.send_text(word)
                await asyncio.sleep(0.2)
            await websocket.send_text("[END]")
    except Exception as e:
        await websocket.send_text(f"Error: {str(e)}")
        await websocket.close()
    finally:
        print("WebSocket connection closed.")



@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")
