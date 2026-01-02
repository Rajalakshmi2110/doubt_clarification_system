"""
Module 4C: Grounded Answer Generation
Generates syllabus-grounded answers using fine-tuned FLAN-T5 model.
"""

import json
import sys
import os
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from transformers import T5ForConditionalGeneration, AutoTokenizer
from modules.module2_semantic_indexing.semantic_indexing import SemanticIndexer

def load_sample_question():
    """Load the same sample question used in Module 4A"""
    test_file = "data/processed/mcp_test.jsonl"
    
    with open(test_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        sample = json.loads(first_line)
    
    return sample

def retrieve_context(question, indexer, k=4):
    """Retrieve relevant context using FAISS index (same as Module 4A)"""
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
    """Create MCP-formatted input (same as Module 4A)"""
    mcp_input = f"""[QUESTION]
{question}

[CONTEXT]
{context}

[INSTRUCTION]
Answer the question academically using only the given context."""
    
    return mcp_input

def generate_grounded_answer(mcp_input, model, tokenizer, max_length=200):
    """Generate answer using fine-tuned FLAN-T5"""
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
    print("MODULE 4C: GROUNDED ANSWER GENERATION")
    print("Using Fine-tuned FLAN-T5 (Syllabus-Grounded)")
    print("=" * 60)
    
    # Load same sample question as Module 4A
    print("\n1. Loading sample question from MCP dataset...")
    sample = load_sample_question()
    question = sample['input'].split('[QUESTION]\n')[1].split('\n\n[CONTEXT]')[0]
    
    print(f"Question: {question}")
    
    # Load FAISS indexer (same as Module 4A)
    print("\n2. Loading FAISS semantic indexer...")
    indexer = SemanticIndexer()
    indexer.load_index()
    
    # Retrieve context (same strategy as Module 4A)
    print("\n3. Retrieving relevant textbook context...")
    context = retrieve_context(question, indexer, k=4)
    
    # Create MCP input (same format as Module 4A)
    mcp_input = create_mcp_input(question, context)
    
    # Load fine-tuned model
    print("\n4. Loading fine-tuned FLAN-T5 model...")
    model_dir = "models/flan_t5_mcp"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    
    # Generate grounded answer
    print("\n5. Generating syllabus-grounded answer...")
    grounded_answer = generate_grounded_answer(mcp_input, model, tokenizer)
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nQUESTION:\n{question}")
    print(f"\nSYLLABUS-GROUNDED ANSWER (Fine-tuned FLAN-T5):\n{grounded_answer}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()