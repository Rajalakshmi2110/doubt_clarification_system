import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
from typing import List, Dict, Any

class SemanticIndexer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.textbook_chunks = []
        self.chunk_metadata = []
        
    def load_textbook_chunks(self, data_dir: str = "data/processed"):
        """Load only textbook chunks (exclude notes and syllabus)"""
        textbook_files = [
            "knowledge_chunks_primary_textbook_clean.json",
            "knowledge_chunks_secondary_textbook_clean.json"
        ]
        
        for file in textbook_files:
            file_path = Path(data_dir) / file
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    self.textbook_chunks.extend([chunk['text'] for chunk in chunks])
                    self.chunk_metadata.extend(chunks)
        
        print(f"Loaded {len(self.textbook_chunks)} textbook chunks")
        return len(self.textbook_chunks)
    
    def generate_embeddings(self):
        """Generate SBERT embeddings for textbook chunks"""
        print("Generating embeddings...")
        embeddings = self.model.encode(self.textbook_chunks, show_progress_bar=True)
        return embeddings.astype('float32')
    
    def create_faiss_index(self, embeddings: np.ndarray):
        """Create FAISS vector index"""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        print(f"Created FAISS index with {self.index.ntotal} vectors")
        return self.index
    
    def save_index(self, index_path: str = "data/processed/textbook_faiss.index", 
                   metadata_path: str = "data/processed/textbook_metadata.pkl"):
        """Save FAISS index and metadata"""
        faiss.write_index(self.index, index_path)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.chunk_metadata, f)
        
        print(f"Saved index to {index_path}")
        print(f"Saved metadata to {metadata_path}")
    
    def load_index(self, index_path: str = "data/processed/textbook_faiss.index",
                   metadata_path: str = "data/processed/textbook_metadata.pkl"):
        """Load existing FAISS index and metadata"""
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, 'rb') as f:
            self.chunk_metadata = pickle.load(f)
        
        print(f"Loaded index with {self.index.ntotal} vectors")
        return self.index
    
    def search(self, query: str, k: int = 5):
        """Search for similar textbook chunks"""
        query_embedding = self.model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunk_metadata):
                result = {
                    'rank': i + 1,
                    'score': float(score),
                    'text': self.chunk_metadata[idx]['text'][:200] + '...',
                    'source': 'textbook',
                    'similarity': f"{float(score):.3f}"
                }
                results.append(result)
        
        return results

def build_textbook_index():
    """Main function to build textbook semantic index"""
    indexer = SemanticIndexer()
    
    # Load textbook chunks only
    chunk_count = indexer.load_textbook_chunks()
    if chunk_count == 0:
        print("No textbook chunks found!")
        return
    
    # Generate embeddings
    embeddings = indexer.generate_embeddings()
    
    # Create FAISS index
    indexer.create_faiss_index(embeddings)
    
    # Save index
    indexer.save_index()
    
    print(f"\nModule 2 Complete: Textbook semantic index created with {chunk_count} chunks")
    return indexer

if __name__ == "__main__":
    build_textbook_index()