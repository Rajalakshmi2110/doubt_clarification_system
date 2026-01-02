"""
Module 4A: Baseline Answer Generation
Generates baseline answers using pretrained FLAN-T5 without fine-tuning.
"""

import json
import sys
import os
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from transformers import T5ForConditionalGeneration, AutoTokenizer
from modules.module2_semantic_indexing.semantic_indexing import SemanticIndexer

def load_sample_question():
    """Load a sample question from MCP test dataset"""
    test_file = "data/processed/mcp_test.jsonl"
    
    with open(test_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        sample = json.loads(first_line)
    
    return sample

def retrieve_context(question, indexer, k=4):
    """Retrieve relevant context using FAISS index"""
    # Get search results with indices
    query_embedding = indexer.model.encode([question]).astype('float32')
    import faiss
    faiss.normalize_L2(query_embedding)
    
    scores, indices = indexer.index.search(query_embedding, k)
    
    context_chunks = []
    for idx in indices[0]:
        if idx < len(indexer.chunk_metadata):
            full_text = indexer.chunk_metadata[idx]['text']
            context_chunks.append(full_text)
    
    return "\n\n".join(context_chunks)

def create_mcp_input(question, context):
    """Create MCP-formatted input"""
    mcp_input = f"""[QUESTION]
{question}

[CONTEXT]
{context}

[INSTRUCTION]
Answer the question academically using only the given context."""
    
    return mcp_input

def generate_baseline_answer(mcp_input, model, tokenizer, max_length=200):
    """Generate answer using pretrained FLAN-T5"""
    inputs = tokenizer(mcp_input, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def main():
    print("=" * 60)
    print("MODULE 4A: BASELINE ANSWER GENERATION")
    print("Using Pretrained FLAN-T5 (No Fine-tuning)")
    print("=" * 60)
    
    # Load sample question
    print("\n1. Loading sample question from MCP dataset...")
    sample = load_sample_question()
    question = sample['input'].split('[QUESTION]\n')[1].split('\n\n[CONTEXT]')[0]
    
    print(f"Question: {question}")
    
    # Load FAISS indexer
    print("\n2. Loading FAISS semantic indexer...")
    indexer = SemanticIndexer()
    indexer.load_index()
    
    # Retrieve context
    print("\n3. Retrieving relevant textbook context...")
    context = retrieve_context(question, indexer, k=4)
    
    # Create MCP input
    mcp_input = create_mcp_input(question, context)
    
    # Load pretrained model
    print("\n4. Loading pretrained FLAN-T5 model...")
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Generate baseline answer
    print("\n5. Generating baseline answer...")
    baseline_answer = generate_baseline_answer(mcp_input, model, tokenizer)
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nQUESTION:\n{question}")
    print(f"\nGENERALIZED ANSWER (Pretrained FLAN-T5):\n{baseline_answer}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()