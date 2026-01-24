# Module 2: Textbook Semantic Indexing

## Purpose
Create a searchable textbook knowledge base using SBERT embeddings and FAISS vector indexing.

## Features
- **SBERT Embeddings**: Uses `all-MiniLM-L6-v2` model for semantic text representation
- **FAISS Vector Index**: Efficient similarity search with cosine similarity
- **Textbook-Only Focus**: Indexes only textbook content (excludes notes/syllabus)
- **Persistent Storage**: Saves index and metadata for reuse

## Usage
```python
from modules.module2_semantic_indexing.semantic_indexing import SemanticIndexer

# Build index
indexer = build_textbook_index()

# Search textbooks
results = indexer.search("TCP protocol", k=5)
```

## Output Files
- `data/processed/textbook_faiss.index` - FAISS vector index
- `data/processed/textbook_metadata.pkl` - Chunk metadata

## Dependencies
- sentence-transformers
- faiss-cpu
- numpy