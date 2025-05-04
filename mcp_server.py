from mcp.server.fastmcp import FastMCP, Image
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent
from mcp import types
import math
import sys
import os
import json
import faiss
import numpy as np
from pathlib import Path
import requests
import time
import pickle
from document_processor import DocumentProcessor, Document
from build_index import IndexBuilder
import logging

mcp = FastMCP("Analyzer")

EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 40
ROOT = Path(__file__).parent.resolve()

def get_embedding(text: str) -> np.ndarray:
    response = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "prompt": text})
    response.raise_for_status()
    return np.array(response.json()["embedding"], dtype=np.float32)

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    for i in range(0, len(words), size - overlap):
        yield " ".join(words[i:i+size])

def mcp_log(level: str, message: str) -> None:
    """Log a message to stderr to avoid interfering with JSON communication"""
    sys.stderr.write(f"{level}: {message}\n")
    sys.stderr.flush()

@mcp.tool()
def search_documents(query: str) -> list[str]:
    """Search for relevant content from uploaded documents."""
    ensure_faiss_ready()
    mcp_log("SEARCH", f"Query: {query}")
    try:
        index = faiss.read_index(str(ROOT / "faiss_index" / "swiggy.index"))
        with open(file_path, 'rb') as file:
            metadata = pickle.load((ROOT / "faiss_index" / "documents.pkl"))
        query_vec = get_embedding(query).reshape(1, -1)
        D, I = index.search(query_vec, k=5)
        results = []
        for idx in I[0]:
            data = metadata[idx]
            results.append(f"{data['content']}\n[Source: {data['metadata']['source']}]")
        return results
    except Exception as e:
        return [f"ERROR: Failed to search: {str(e)}"]

# DEFINE RESOURCES

# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    print("CALLED: get_greeting(name: str) -> str:")
    return f"Hello, {name}!"


# DEFINE AVAILABLE PROMPTS
@mcp.prompt()
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"
    print("CALLED: review_code(code: str) -> str:")


@mcp.prompt()
def debug_error(error: str) -> list[base.Message]:
    return [
        base.UserMessage("I'm seeing this error:"),
        base.UserMessage(error),
        base.AssistantMessage("I'll help debug that. What have you tried so far?"),
    ]

@mcp.prompt()
def analyze_statement(question: str, context: list[str]) -> str:
    """Analyze Swiggy statement based on context and question."""
    context_text = "\n".join(context)
    return get_llm_response(question, context_text)

@mcp.prompt()
def summarize_orders(context: list[str]) -> str:
    """Generate a summary of orders from the statement."""
    question = "Please provide a summary of all orders including total amount spent, number of orders, and most ordered items."
    return get_llm_response(question, "\n".join(context))

@mcp.prompt()
def analyze_spending_patterns(context: list[str]) -> str:
    """Analyze spending patterns from the statement."""
    question = "Please analyze the spending patterns including average order value, peak ordering times, and spending trends."
    return get_llm_response(question, "\n".join(context)) 

def process_documents(pdf_dir = "data/"):
    """Process documents and create FAISS index"""
    mcp_log("INFO", "Indexing documents...")
    
    pdf_files = [
        f for f in os.listdir(pdf_dir) 
        if f.endswith('.pdf')
    ]
    
    if not pdf_files:
        mcp_log("ERROR","No PDF files found in the current directory")
        return
        
    pdf_paths = [os.path.join(pdf_dir, f) for f in pdf_files]
    
    try:
        # Initialize and build index
        builder = IndexBuilder()
        builder.process_and_embed_documents(pdf_paths)
        
        # Save the index and related data
        builder.save_index()
        mcp_log("INFO","Index building completed successfully")
        
    except Exception as e:
        mcp_log("ERROR",f"Error building index: {str(e)}")
        raise

def ensure_faiss_ready():
    from pathlib import Path
    index_path = ROOT / "faiss_index" / "swiggy.index"
    meta_path = ROOT / "faiss_index" / "documents.pkl"
    if not (index_path.exists() and meta_path.exists()):
        mcp_log("INFO", "Index not found — running process_documents()...")
        process_documents()
    else:
        mcp_log("INFO", "Index already exists. Skipping regeneration.")


if __name__ == "__main__":
    print("STARTING THE SERVER AT AMAZING LOCATION")

    
    
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        mcp.run() # Run without transport for dev server
    else:
        # Start the server in a separate thread
        import threading
        server_thread = threading.Thread(target=lambda: mcp.run(transport="stdio"))
        server_thread.daemon = True
        server_thread.start()
        
        # Wait a moment for the server to start
        time.sleep(2)
        
        # Process documents after server is running if not done already
        # process_documents()
        ensure_faiss_ready()
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
