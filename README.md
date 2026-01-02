# Academic Doubt Clarification System

A RAG-based system for answering academic questions using textbook knowledge and semantic search.

## Project Status

✅ **Module 1**: Academic Knowledge Ingestion & Preparation - COMPLETE
- PDF text extraction and cleaning
- Content preprocessing (removed prefaces, copyright, indexes)
- Semantic chunking (~400 chars, sentence boundaries)
- **6,866 total knowledge chunks** processed

✅ **Module 2**: Textbook Semantic Indexing (SBERT + FAISS) - COMPLETE
- SBERT embeddings (all-MiniLM-L6-v2, 384-dim)
- FAISS vector index with cosine similarity
- **6,850 textbook chunks** indexed
- Semantic search API with 0.6-0.8 similarity scores

## Architecture

```
Module 1: Knowledge Ingestion → Module 2: Semantic Indexing → Module 3: Q&A System
```

## Knowledge Base
- **Primary Textbook**: Computer Networking A Top-Down Approach (2,691 chunks)
- **Secondary Textbook**: Data and Computer Communications by William Stallings (4,159 chunks)
- **Course Notes**: 6 chunks
- **Syllabus**: 10 chunks (metadata only, not indexed)

## Technical Stack
- **Embeddings**: Sentence-BERT (sentence-transformers)
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **Processing**: PyMuPDF, Python, Regex
- **Format**: JSON knowledge chunks with metadata

## Usage

### Test Semantic Search
```bash
python test_module2.py
```

### Interactive Search
```python
from modules.module2_semantic_indexing.semantic_indexing import SemanticIndexer

indexer = SemanticIndexer()
indexer.load_index()
results = indexer.search("TCP protocol", k=5)
```

## Files Structure
```
├── modules/
│   ├── module1_knowledge_ingestion/
│   └── module2_semantic_indexing/
├── data/
│   ├── raw/ (not included - large PDFs)
│   └── processed/ (JSON chunks, FAISS index)
├── requirements.txt
└── test_module2.py
```

## Next Steps
- Module 3: Question-Answering System
- Module 4: Query Processing & Response Generation
- Module 5: Web Interface & Deployment