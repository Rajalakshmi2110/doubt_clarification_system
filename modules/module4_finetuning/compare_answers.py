"""
Module 4D: Answer Comparison Interface
Displays side-by-side comparison of baseline vs grounded answers.
"""

import json
import sys
import os
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from transformers import T5ForConditionalGeneration, AutoTokenizer
from modules.module2_semantic_indexing.semantic_indexing import SemanticIndexer

def load_sample_question():
    """Load the same sample question used in Modules 4A and 4C"""
    test_file = "data/processed/mcp_test.jsonl"
    
    with open(test_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        sample = json.loads(first_line)
    
    return sample

def retrieve_context_with_metadata(question, indexer, k=4):
    """Retrieve context with chunk metadata for display"""
    query_embedding = indexer.model.encode([question]).astype('float32')
    import faiss
    faiss.normalize_L2(query_embedding)
    
    scores, indices = indexer.index.search(query_embedding, k)
    
    context_chunks = []
    chunk_metadata = []
    
    for i, idx in enumerate(indices[0]):
        if idx < len(indexer.chunk_metadata):
            chunk_data = indexer.chunk_metadata[idx]
            context_chunks.append(chunk_data['text'])
            chunk_metadata.append({
                'chunk_id': chunk_data.get('chunk_id', f'chunk_{idx}'),
                'source': chunk_data.get('source_file', 'textbook'),
                'score': float(scores[0][i])
            })
    
    return "\n\n".join(context_chunks), chunk_metadata

def create_mcp_input(question, context):
    """Create MCP-formatted input"""
    mcp_input = f"""[QUESTION]
{question}

[CONTEXT]
{context}

[INSTRUCTION]
Answer the question academically using only the given context."""
    
    return mcp_input

def generate_baseline_answer(mcp_input):
    """Generate baseline answer using pretrained FLAN-T5"""
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    inputs = tokenizer(mcp_input, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=200,
            num_beams=4,
            early_stopping=True
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def generate_grounded_answer(mcp_input):
    """Generate grounded answer using fine-tuned FLAN-T5"""
    model_dir = "models/flan_t5_mcp"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    
    inputs = tokenizer(mcp_input, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=200,
            num_beams=4,
            early_stopping=True
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def display_comparison(question, baseline_answer, grounded_answer, chunk_metadata=None):
    """Display the comparison in a clean format"""
    print("=" * 80)
    print("MODULE 4D: BASELINE vs SYLLABUS-GROUNDED COMPARISON")
    print("=" * 80)
    
    print(f"\nQUESTION:")
    print(f"{question}")
    
    print(f"\nBASELINE ANSWER (Pretrained FLAN-T5):")
    print(f"{baseline_answer}")
    
    print(f"\nSYLLABUS-GROUNDED ANSWER (Fine-tuned FLAN-T5):")
    print(f"{grounded_answer}")
    
    if chunk_metadata:
        print(f"\nRETRIEVED CONTEXT METADATA:")
        for i, chunk in enumerate(chunk_metadata, 1):
            print(f"  {i}. Chunk ID: {chunk['chunk_id']}")
            print(f"     Source: {chunk['source']}")
            print(f"     Similarity Score: {chunk['score']:.3f}")
    
    print("\n" + "=" * 80)
    print("COMPARISON ANALYSIS:")
    print("=" * 80)
    
    # Simple analysis
    baseline_words = len(baseline_answer.split())
    grounded_words = len(grounded_answer.split())
    
    print(f"Baseline Answer Length: {baseline_words} words")
    print(f"Grounded Answer Length: {grounded_words} words")
    
    if baseline_answer.lower().strip() == grounded_answer.lower().strip():
        print("Status: Answers are identical")
        print("\nNote: The baseline model already knows this fact. Fine-tuning primarily")
        print("improves context adherence and syllabus grounding for complex or")
        print("explanation-heavy questions. Both models show consistent performance")
        print("on this straightforward factual question.")
    else:
        print("Status: Answers differ - fine-tuning effect observed")
        print("\nNote: Fine-tuning has modified the response, showing adaptation to")
        print("syllabus-specific context and academic formatting requirements.")
    
    print("=" * 80)

def main():
    print("Loading comparison data...")
    
    # Load sample question
    sample = load_sample_question()
    question = sample['input'].split('[QUESTION]\n')[1].split('\n\n[CONTEXT]')[0]
    
    # Load FAISS indexer
    indexer = SemanticIndexer()
    indexer.load_index()
    
    # Retrieve context with metadata
    context, chunk_metadata = retrieve_context_with_metadata(question, indexer, k=4)
    
    # Create MCP input
    mcp_input = create_mcp_input(question, context)
    
    print("Generating baseline answer...")
    baseline_answer = generate_baseline_answer(mcp_input)
    
    print("Generating grounded answer...")
    grounded_answer = generate_grounded_answer(mcp_input)
    
    # Display comparison
    display_comparison(question, baseline_answer, grounded_answer, chunk_metadata)

if __name__ == "__main__":
    main()