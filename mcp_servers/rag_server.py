"""
mcp_servers/rag_server.py
Local RAG Engine — handles vector embeddings (via Ollama nomic-embed-text)
and semantic search (via ChromaDB).
"""
import os
import chromadb
from pathlib import Path
from typing import Optional, List
import ollama

class RAGServer:
    """
    RAG Server wrapping ChromaDB to provide semantic search
    over the local corpus using Ollama embeddings.
    """
    
    DB_PATH = Path(__file__).parent.parent / "data" / "chroma_db"
    COLLECTION_NAME = "papers_corpus"
    EMBEDDING_MODEL = "nomic-embed-text"

    def __init__(self):
        # Create persistent ChromaDB client
        self.DB_PATH.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.DB_PATH))
        self.collection = self.client.get_or_create_collection(name=self.COLLECTION_NAME)

    def _get_embedding(self, text: str) -> list[float]:
        """Generate vector embedding using local Ollama model."""
        response = ollama.embeddings(model=self.EMBEDDING_MODEL, prompt=text)
        return response['embedding']

    def add_chunks(self, texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
        """
        Embed and store document chunks in ChromaDB.
        """
        if not texts:
            return
            
        print(f"[RAGServer] Generating embeddings for {len(texts)} chunks using {self.EMBEDDING_MODEL}...")
        embeddings = [self._get_embedding(t) for t in texts]
        
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"[RAGServer] Added {len(texts)} chunks to ChromaDB.")

    def query(self, query_text: str, top_k: int = 3) -> list[str]:
        """
        Query the local corpus for semantically similar chunks.
        Returns the raw document texts.
        """
        if self.collection.count() == 0:
            print("[RAGServer] Warning: ChromaDB collection is empty. Returning no results.")
            return []
            
        print(f"[RAGServer] Querying internal corpus for: '{query_text}'")
        query_embedding = self._get_embedding(query_text)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        documents = results.get("documents", [[]])[0]
        return documents
