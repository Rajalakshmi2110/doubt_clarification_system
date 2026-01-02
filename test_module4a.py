"""
Test Module 4A: Baseline Answer Generation
Validates the baseline inference functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.module4_finetuning.baseline_inference import (
    load_sample_question, 
    retrieve_context, 
    create_mcp_input,
    generate_baseline_answer
)
from modules.module2_semantic_indexing.semantic_indexing import SemanticIndexer
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

def test_sample_loading():
    """Test loading sample question from MCP dataset"""
    print("Testing sample question loading...")
    sample = load_sample_question()
    
    assert 'input' in sample, "Sample should contain 'input' field"
    assert '[QUESTION]' in sample['input'], "Input should contain [QUESTION] section"
    assert '[CONTEXT]' in sample['input'], "Input should contain [CONTEXT] section"
    assert '[INSTRUCTION]' in sample['input'], "Input should contain [INSTRUCTION] section"
    
    print("[PASS] Sample loading test passed")
    return sample

def test_context_retrieval():
    """Test FAISS context retrieval"""
    print("Testing context retrieval...")
    
    indexer = SemanticIndexer()
    indexer.load_index()
    
    test_question = "What is TCP protocol?"
    context = retrieve_context(test_question, indexer, k=2)
    
    assert len(context) > 0, "Context should not be empty"
    assert isinstance(context, str), "Context should be a string"
    
    print("[PASS] Context retrieval test passed")
    return context

def test_mcp_formatting():
    """Test MCP input formatting"""
    print("Testing MCP input formatting...")
    
    question = "Test question?"
    context = "Test context content."
    mcp_input = create_mcp_input(question, context)
    
    assert '[QUESTION]' in mcp_input, "MCP input should contain [QUESTION]"
    assert '[CONTEXT]' in mcp_input, "MCP input should contain [CONTEXT]"
    assert '[INSTRUCTION]' in mcp_input, "MCP input should contain [INSTRUCTION]"
    assert question in mcp_input, "MCP input should contain the question"
    assert context in mcp_input, "MCP input should contain the context"
    
    print("[PASS] MCP formatting test passed")
    return mcp_input

def test_model_loading():
    """Test FLAN-T5 model loading"""
    print("Testing model loading...")
    
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    assert tokenizer is not None, "Tokenizer should load successfully"
    assert model is not None, "Model should load successfully"
    
    print("[PASS] Model loading test passed")
    return model, tokenizer

def test_answer_generation():
    """Test baseline answer generation"""
    print("Testing answer generation...")
    
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    test_input = "[QUESTION]\nWhat is a network protocol?\n\n[CONTEXT]\nA protocol defines the format and the order of messages exchanged between two or more communicating entities.\n\n[INSTRUCTION]\nAnswer the question academically using only the given context."
    
    answer = generate_baseline_answer(test_input, model, tokenizer, max_length=50)
    
    assert len(answer) > 0, "Answer should not be empty"
    assert isinstance(answer, str), "Answer should be a string"
    
    print("[PASS] Answer generation test passed")
    print(f"Generated answer: {answer}")
    return answer

def run_all_tests():
    """Run all Module 4A tests"""
    print("=" * 50)
    print("MODULE 4A VALIDATION TESTS")
    print("=" * 50)
    
    try:
        # Test 1: Sample loading
        sample = test_sample_loading()
        
        # Test 2: Context retrieval
        context = test_context_retrieval()
        
        # Test 3: MCP formatting
        mcp_input = test_mcp_formatting()
        
        # Test 4: Model loading
        model, tokenizer = test_model_loading()
        
        # Test 5: Answer generation
        answer = test_answer_generation()
        
        print("\nALL TESTS PASSED [SUCCESS]")
        print("Module 4A is working correctly!")
        
    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        print("=" * 50)
        raise

if __name__ == "__main__":
    run_all_tests()