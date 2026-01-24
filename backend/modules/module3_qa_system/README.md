# Module 3: Context-Aware Q&A System

## Overview
Module 3 creates a supervised training dataset by augmenting existing Question-Answer pairs with retrieved textbook context using the FAISS index from Module 2.

## Architecture
```
Q&A Dataset → FAISS Retrieval → MCP Format → Train/Val/Test Splits
```

## Key Components

### 1. MCPDatasetGenerator
- **Purpose**: Generate Model Context Protocol (MCP) style training data
- **Input**: Academic Q&A pairs + FAISS textbook index
- **Output**: Context-augmented training examples

### 2. MCP Format Structure
```
[QUESTION]
<question text>

[CONTEXT]
<concatenated retrieved textbook chunks>

[INSTRUCTION]
Answer the question academically using only the given context.
```

### 3. Dataset Schema
```json
{
  "input": "MCP formatted input with question, context, and instruction",
  "output": "Gold-standard academic answer",
  "metadata": {
    "question_id": 1,
    "topic": "Network Hardware",
    "difficulty": "medium",
    "retrieved_chunks": [
      {
        "chunk_id": "chunk_123",
        "source_file": "primary_textbook",
        "unit": "Data Link Layer",
        "score": 0.85
      }
    ]
  }
}
```

## Usage

### Generate MCP Dataset
```python
from modules.module3_qa_system.mcp_dataset_generator import MCPDatasetGenerator

# Initialize generator
generator = MCPDatasetGenerator()

# Generate complete dataset
mcp_dataset = generator.generate_mcp_dataset()

# Split and save (80:10:10)
summary = generator.split_and_save_dataset(mcp_dataset)
```

### Test Generation
```bash
python test_module3.py
```

## Output Files
- `data/processed/mcp_train.jsonl` - Training set (80%)
- `data/processed/mcp_val.jsonl` - Validation set (10%)  
- `data/processed/mcp_test.jsonl` - Test set (10%)
- `data/processed/mcp_dataset_summary.json` - Dataset statistics

## Key Design Decisions

1. **Retrieval Strategy**: Top-k=4 chunks per question for balanced context
2. **Context Source**: ONLY textbook chunks (excludes notes/syllabus)
3. **Format**: MCP-style for clear instruction following
4. **Splits**: Standard 80:10:10 for ML training
5. **Metadata**: Rich tracking for analysis and debugging

## Dependencies
- sentence-transformers (SBERT model)
- faiss-cpu (vector search)
- scikit-learn (dataset splitting)
- numpy (numerical operations)

## Integration
- **Input**: Module 2 FAISS index + Q&A dataset
- **Output**: Ready for T5/FLAN-T5 fine-tuning
- **Next**: Module 4 (Query Processing & Response Generation)