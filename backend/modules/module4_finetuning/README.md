# Module 4A: Baseline Answer Generation

## Overview
Module 4A generates baseline answers using a **pretrained FLAN-T5 model without any fine-tuning**. This serves as a control to demonstrate the improvement achieved through fine-tuning in later modules.

## Purpose
- Generate "generalized answers" using vanilla FLAN-T5
- Establish baseline performance before fine-tuning
- Demonstrate the need for academic context grounding

## Implementation

### Files
- `baseline_inference.py` - Main script for baseline answer generation

### Process
1. **Load Sample Question**: Extracts a question from the MCP test dataset
2. **Retrieve Context**: Uses FAISS index to get relevant textbook chunks (top-k=4)
3. **Format Input**: Creates MCP-formatted input with question, context, and instruction
4. **Generate Answer**: Uses pretrained FLAN-T5-small to generate response
5. **Display Results**: Shows question and generalized answer

### Model Used
- **google/flan-t5-small**: Pretrained model (248M parameters)
- **No fine-tuning**: Uses model as-is from Hugging Face

## Usage

```bash
python modules/module4_finetuning/baseline_inference.py
```

## Expected Output
```
QUESTION:
What is the main difference between fiber optic cables and twisted pair cables in terms of data transmission?

GENERALIZED ANSWER (Pretrained FLAN-T5):
[Generated baseline answer without academic grounding]
```

## Dependencies
- transformers
- torch
- sentencepiece
- FAISS index from Module 2
- MCP dataset from Module 3

## Next Steps
- Module 4B: Fine-tune FLAN-T5 on MCP dataset
- Module 4C: Generate grounded answers with fine-tuned model
- Module 4D: Compare baseline vs grounded answers