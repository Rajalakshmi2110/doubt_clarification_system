# Academic Doubt Clarification System - Backend

Python Flask API backend for the Academic Doubt Clarification System.

## Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r api_requirements.txt
```

### 2. Start API Server
```bash
python api_server.py
```
Backend runs on: http://localhost:8000

## API Endpoints

- `POST /api/validate`: Validate a question
- `GET /api/health`: Check system health

## Project Structure

```
backend/
├── api_server.py           # Flask API server
├── api_requirements.txt    # API dependencies
├── requirements.txt        # Core ML dependencies
├── config.json            # Configuration
├── data/                  # Dataset and processed files
├── models/                # Fine-tuned models
├── modules/               # Core ML modules
└── tests/                 # Test files
```

## Technology Stack

- **API**: Flask with CORS support
- **ML Models**: FLAN-T5, SentenceTransformers
- **Vector Search**: FAISS
- **NLP**: spaCy, scikit-learn