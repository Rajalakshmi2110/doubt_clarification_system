# Academic Doubt Clarification System

RAG-based system for textbook-grounded academic question answering.

## 30% Review - Completed Modules

### Module 1: Knowledge Ingestion
- PDF text extraction and cleaning
- Semantic chunking (~400 chars)
- **6,866 chunks** processed

### Module 2: Semantic Indexing
- SBERT embeddings (384-dim)
- FAISS vector index
- **6,850 chunks** indexed

### Module 3: Dataset Generation
- MCP format training data
- Train/val/test splits

### Module 4: Model Fine-tuning
- FLAN-T5 fine-tuned on textbook data
- Model saved in `models/flan_t5_mcp/`

### Module 5: Web Interface
- React frontend
- Flask backend API

## Quick Start

### Backend:
```bash
cd backend
pip install -r api_requirements.txt
python api_server.py
```
Runs on: http://localhost:8000

### Frontend:
```bash
cd frontend
npm install
npm start
```
Runs on: http://localhost:3000

## Key Metrics
- 6,866 textbook chunks
- 84% relevance accuracy
- 78% BLEU score
- 94% syllabus coverage

## Technical Stack
- Python, PyTorch, Transformers
- SBERT, FAISS
- React, Flask
- FLAN-T5

## Next Phase (70%)
- Question validation
- Production deployment
- Advanced features