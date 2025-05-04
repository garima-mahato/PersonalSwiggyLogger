import os
import json
import numpy as np
import faiss
import requests
from typing import List, Dict, Any, Tuple
from document_processor import DocumentProcessor, Document
import logging
import pickle

# Configure logging
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler("logs/build_index.log")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class IndexBuilder:
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: List[Document] = []
        self.embeddings: List[np.ndarray] = []
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embeddings from local Ollama server using nomic-embed-text model."""
        try:
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={
                    "model": "nomic-embed-text",
                    "prompt": text
                }
            )
            response.raise_for_status()
            return np.array(response.json()["embedding"], dtype=np.float32)
        except Exception as e:
            logger.error(f"Error getting embedding from Ollama: {str(e)}")
            raise

    def process_and_embed_documents(self, pdf_files: List[str]) -> None:
        """Process PDF files and create embeddings."""
        try:
            doc_processor = DocumentProcessor()
            
            for pdf_file in pdf_files:
                logger.info(f"Processing PDF file: {pdf_file}")
                documents = doc_processor.process_pdf(pdf_file)
                
                for doc in documents:
                    try:
                        embedding = self.get_embedding(doc.content)
                        self.documents.append(doc)
                        self.embeddings.append(embedding)
                        logger.info(f"Created embedding for document from {doc.metadata['source']}")
                    except Exception as e:
                        logger.error(f"Error creating embedding for document: {str(e)}")
                        continue
            
            if self.embeddings:
                embeddings_array = np.stack(self.embeddings)#.astype('float32')
                self.index.add(embeddings_array)
                logger.info("Successfully added all embeddings to FAISS index")
                
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise

    def save_index(self, index_dir: str = "faiss_index") -> None:
        """Save the FAISS index and related data."""
        try:
            os.makedirs(index_dir, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, os.path.join(index_dir, "swiggy.index"))
            
            # Save documents and their metadata
            documents_data = [
                {
                    "content": doc.content,
                    "metadata": doc.metadata
                }
                for doc in self.documents
            ]
            
            with open(os.path.join(index_dir, "documents.pkl"), "wb") as f:
                pickle.dump(documents_data, f)
                
            # Save embeddings
            np.save(os.path.join(index_dir, "embeddings.npy"), np.array(self.embeddings))
            
            logger.info(f"Successfully saved index and data to {index_dir}")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise

def main():
    # Directory containing PDF files
    pdf_dir = "data/"
    pdf_files = [
        f for f in os.listdir(pdf_dir) 
        if f.endswith('.pdf')
    ]
    
    if not pdf_files:
        logger.error("No PDF files found in the current directory")
        return
        
    pdf_paths = [os.path.join(pdf_dir, f) for f in pdf_files]
    
    try:
        # Initialize and build index
        builder = IndexBuilder()
        builder.process_and_embed_documents(pdf_paths)
        
        # Save the index and related data
        builder.save_index()
        logger.info("Index building completed successfully")
        
    except Exception as e:
        logger.error(f"Error building index: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
