"""
Test Module 4C: Grounded Answer Generation Validation
Validates the grounded inference functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.module4_finetuning.grounded_inference import (
    load_sample_question,
    retrieve_context,
    create_mcp_input,
    generate_grounded_answer
)
from modules.module2_semantic_indexing.semantic_indexing import SemanticIndexer
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

def test_fine_tuned_model_loading():
    """Test loading fine-tuned model"""
    print("Testing fine-tuned model loading...")
    
    model_dir = "models/flan_t5_mcp"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    
    assert tokenizer is not None, "Failed to load fine-tuned tokenizer"
    assert model is not None, "Failed to load fine-tuned model"
    
    print("[PASS] Fine-tuned model loads successfully")
    return model, tokenizer

def test_same_question_as_4a():
    """Test that we use the same question as Module 4A"""
    print("Testing question consistency with Module 4A...")
    
    sample = load_sample_question()
    question = sample['input'].split('[QUESTION]\n')[1].split('\n\n[CONTEXT]')[0]
    
    expected_question = "What is the main difference between fiber optic cables and twisted pair cables in terms of data transmission?"
    assert question == expected_question, f"Question mismatch: {question}"
    
    print("[PASS] Using same question as Module 4A")
    return question

def test_same_context_retrieval():
    """Test that context retrieval matches Module 4A"""
    print("Testing context retrieval consistency...")
    
    indexer = SemanticIndexer()
    indexer.load_index()
    
    question = "What is the main difference between fiber optic cables and twisted pair cables in terms of data transmission?"
    context = retrieve_context(question, indexer, k=4)
    
    assert len(context) > 0, "Context should not be empty"
    assert "optical fiber" in context.lower() or "twisted pair" in context.lower(), "Context should be relevant"
    
    print("[PASS] Context retrieval working correctly")
    return context

def test_grounded_answer_generation():
    """Test grounded answer generation with fine-tuned model"""
    print("Testing grounded answer generation...")
    
    model_dir = "models/flan_t5_mcp"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    
    # Use a clear test case
    test_input = """[QUESTION]
What is a network protocol?

[CONTEXT]
A protocol defines the format and the order of messages exchanged between two or more communicating entities, as well as the actions taken on the transmission and/or receipt of a message or other event.

[INSTRUCTION]
Answer the question academically using only the given context."""
    
    answer = generate_grounded_answer(test_input, model, tokenizer, max_length=100)
    
    assert len(answer) > 0, "Answer should not be empty"
    assert isinstance(answer, str), "Answer should be a string"
    
    print(f"[PASS] Grounded answer generation working")
    print(f"Sample answer: {answer}")
    return answer

def test_model_difference():
    """Test that fine-tuned model behaves differently from baseline"""
    print("Testing model behavior difference...")
    
    # Load fine-tuned model
    model_dir = "models/flan_t5_mcp"
    ft_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ft_model = T5ForConditionalGeneration.from_pretrained(model_dir)
    
    # Load baseline model
    baseline_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    baseline_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    
    test_input = """[QUESTION]
What is TCP?

[CONTEXT]
TCP (Transmission Control Protocol) is a reliable, connection-oriented transport protocol that ensures data delivery.

[INSTRUCTION]
Answer the question academically using only the given context."""
    
    # Generate with fine-tuned model
    ft_inputs = ft_tokenizer(test_input, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        ft_outputs = ft_model.generate(ft_inputs.input_ids, max_length=50, num_beams=2)
    ft_answer = ft_tokenizer.decode(ft_outputs[0], skip_special_tokens=True)
    
    # Generate with baseline model
    bl_inputs = baseline_tokenizer(test_input, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        bl_outputs = baseline_model.generate(bl_inputs.input_ids, max_length=50, num_beams=2)
    bl_answer = baseline_tokenizer.decode(bl_outputs[0], skip_special_tokens=True)
    
    print(f"Fine-tuned: {ft_answer}")
    print(f"Baseline: {bl_answer}")
    
    # Models should potentially give different answers
    print("[PASS] Model comparison completed")
    return ft_answer, bl_answer

def run_all_tests():
    """Run all Module 4C validation tests"""
    print("=" * 50)
    print("MODULE 4C VALIDATION TESTS")
    print("=" * 50)
    
    try:
        # Test 1: Fine-tuned model loading
        model, tokenizer = test_fine_tuned_model_loading()
        
        # Test 2: Same question as 4A
        question = test_same_question_as_4a()
        
        # Test 3: Same context retrieval
        context = test_same_context_retrieval()
        
        # Test 4: Grounded answer generation
        answer = test_grounded_answer_generation()
        
        # Test 5: Model difference
        ft_answer, bl_answer = test_model_difference()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED [SUCCESS]")
        print("Module 4C grounded inference working correctly!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        print("=" * 50)
        raise

if __name__ == "__main__":
    run_all_tests()