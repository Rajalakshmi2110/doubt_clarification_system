# Academic Doubt Clarification System - Module 1

## Knowledge Ingestion & Preparation

This module handles the extraction, cleaning, and chunking of academic content from PDFs.

### Features:
- PDF text extraction using PyMuPDF
- Noise removal (headers, footers, page numbers)
- Semantic paragraph-based chunking
- Unit assignment based on syllabus keywords
- Metadata attachment for each chunk

### Usage:
```python
from modules.module1_knowledge_ingestion.knowledge_ingestion import KnowledgeIngestionPipeline

# Initialize with syllabus
pipeline = KnowledgeIngestionPipeline("data/raw/syllabus/syllabus.pdf")

# Process textbooks
chunks = pipeline.process_textbook("data/raw/textbooks/primary_textbook.pdf", priority=1)

# Save processed chunks
pipeline.save_chunks(chunks, "data/processed/knowledge_chunks.json")
```

### Data Structure:
Each chunk contains:
- `text`: Cleaned paragraph content
- `unit`: Assigned academic unit
- `source_type`: "textbook" or "notes"
- `book_priority`: 1=primary, 2=secondary, 0=notes
- `source_file`: Original filename
- `chunk_id`: Unique identifier

### Next Steps:
Module 1 prepares the foundation for Module 2 (Semantic Indexing) by providing clean, structured academic content chunks.