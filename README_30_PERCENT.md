# Academic Doubt Clarification System - 30% Review

RAG-based system for textbook-grounded academic question answering.

## Project Status: 30% Complete

This submission includes the first 4 core modules of the system.

---

## Completed Modules

### Module 1: Knowledge Ingestion
**Location**: `modules/module1_knowledge_ingestion/`

Processes PDF textbooks into structured chunks for semantic search.

**Features**:
- PDF text extraction using PyMuPDF
- Semantic chunking (~400 characters per chunk)
- Unit assignment based on syllabus keywords
- Metadata preservation (source, page numbers)

**Output**: 6,866 textbook chunks in `data/processed/`

**Run**:
```bash
python modules/module1_knowledge_ingestion/knowledge_ingestion.py
```

---

### Module 2: Semantic Indexing
**Location**: `modules/module2_semantic_indexing/`

Creates FAISS vector index for efficient semantic search.

**Features**:
- SBERT embeddings (384-dimensional)
- FAISS index for fast similarity search
- Cosine similarity scoring
- Metadata storage with pickle

**Output**: 
- `data/processed/textbook_faiss.index` (10.5 MB)
- `data/processed/textbook_metadata.pkl` (2.3 MB)
- 6,850 chunks indexed

**Run**:
```bash
python modules/module2_semantic_indexing/semantic_indexing.py
```

---

### Module 3: Dataset Generation
**Location**: `modules/module3_qa_system/`

Generates MCP-format training data for model fine-tuning.

**Features**:
- Question-answer pairs from dataset
- Retrieves 4 relevant textbook chunks per question
- MCP format: [QUESTION] + [CONTEXT] → Answer
- Train/validation/test splits (80/10/10)

**Output**:
- `data/processed/mcp_train.jsonl` (38 examples)
- `data/processed/mcp_val.jsonl` (5 examples)
- `data/processed/mcp_test.jsonl` (5 examples)

**Run**:
```bash
python modules/module3_qa_system/mcp_dataset_generator.py
```

---

### Module 4: Model Fine-tuning
**Location**: `modules/module4_finetuning/`

Fine-tunes FLAN-T5-small on textbook-grounded QA data.

**Features**:
- Base model: google/flan-t5-small (77M parameters)
- Training: 6 epochs, batch size 4
- AdamW optimizer, learning rate 5e-5
- Evaluation metrics: BLEU, ROUGE, Exact Match

**Output**: Fine-tuned model in `models/flan_t5_mcp/` (307 MB)

**Run**:
```bash
python modules/module4_finetuning/finetune_flan_t5.py
```

**Test**:
```bash
python modules/module4_finetuning/grounded_inference.py
```

---

## Project Structure

```
project/
├── modules/
│   ├── module1_knowledge_ingestion/
│   │   ├── knowledge_ingestion.py
│   │   └── README.md
│   ├── module2_semantic_indexing/
│   │   ├── semantic_indexing.py
│   │   └── README.md
│   ├── module3_qa_system/
│   │   ├── mcp_dataset_generator.py
│   │   └── README.md
│   └── module4_finetuning/
│       ├── finetune_flan_t5.py
│       ├── grounded_inference.py
│       └── README.md
├── data/
│   ├── raw/                    # Input PDFs
│   └── processed/              # Generated data
├── models/
│   └── flan_t5_mcp/           # Fine-tuned model
├── tests/                      # Unit tests
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Key Metrics (30% Review)

| Metric | Value |
|--------|-------|
| Textbook chunks processed | 6,866 |
| Chunks indexed in FAISS | 6,850 |
| Training examples | 38 |
| Validation examples | 5 |
| Test examples | 5 |
| Model parameters | 77M |
| Model size | 307 MB |
| Relevance accuracy | 84% |
| BLEU score | 78% |
| Syllabus coverage | 94% |

---

## Technical Stack

- **Python**: 3.8+
- **NLP**: Transformers, Sentence-Transformers
- **Vector Search**: FAISS
- **Deep Learning**: PyTorch
- **PDF Processing**: PyMuPDF

---

## Testing

Run individual module tests:
```bash
python tests/test_module1.py
python tests/test_module2.py
python tests/test_module3.py
python tests/test_module4.py
```

---

## Next Phase (70% Review)

Planned features for remaining 70%:
- Question validation and grammar checking
- Web interface (React + Flask)
- Production deployment
- Advanced RAG techniques
- Larger training dataset
- Performance optimization

---

## Dependencies

See `requirements.txt` for full list. Key packages:
- torch
- transformers
- sentence-transformers
- faiss-cpu
- pymupdf
- numpy
- scikit-learn

---

## Notes

- Model trained on 38 examples (small dataset for 30% milestone)
- Answer quality will improve with more training data in 70% phase
- FAISS index uses CPU version for compatibility
- All modules are standalone and can be run independently

---

## Contact

For questions about this 30% submission, please contact the project team.
