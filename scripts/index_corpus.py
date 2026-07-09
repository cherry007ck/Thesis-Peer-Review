"""
scripts/index_corpus.py
Reads the local dataset of papers, chunks their sections,
and inserts them into the ChromaDB vector database.
"""

import sys
import os
import uuid
# Add parent dir to path so we can import from review_agents and mcp_servers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_servers.document_server import DocumentServer
from mcp_servers.rag_server import RAGServer

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Simple sliding window chunker."""
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        i += chunk_size - overlap
    return chunks

def index_all_papers():
    doc_server = DocumentServer()
    rag_server = RAGServer()

    papers = doc_server.list_all_papers()
    if not papers:
        print("No papers found in DocumentServer to index.")
        return

    print(f"Found {len(papers)} papers. Beginning indexing...")

    texts = []
    metadatas = []
    ids = []

    for paper in papers:
        for sec_idx, section in enumerate(paper.sections):
            # Include paper title and section title for better embedding context
            base_text = f"Paper: {paper.title}\nSection: {section.title}\n\n{section.content}"
            
            section_chunks = chunk_text(base_text)
            
            for chunk_idx, chunk in enumerate(section_chunks):
                texts.append(chunk)
                metadatas.append({
                    "paper_id": paper.id,
                    "paper_title": paper.title,
                    "section_title": section.title
                })
                ids.append(f"{paper.id}_sec{sec_idx}_chunk{chunk_idx}")

    print(f"Created {len(texts)} chunks across {len(papers)} papers.")
    
    # Process in batches to avoid overloading the local embedder
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        rag_server.add_chunks(batch_texts, batch_metas, batch_ids)
        print(f"Indexed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

    print("Indexing complete!")

if __name__ == "__main__":
    index_all_papers()
