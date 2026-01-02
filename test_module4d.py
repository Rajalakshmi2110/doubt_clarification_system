"""
Test Module 4D: Answer Comparison Interface Validation
Validates the comparison interface functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.module4_finetuning.compare_answers import (
    load_sample_question,
    retrieve_context_with_metadata,
    create_mcp_input,
    generate_baseline_answer,
    generate_grounded_answer,
    display_comparison
)
from modules.module2_semantic_indexing.semantic_indexing import SemanticIndexer

def test_question_loading():
    """Test loading the same question used in 4A and 4C"""
    print("Testing question loading...")
    
    sample = load_sample_question()
    question = sample['input'].split('[QUESTION]\n')[1].split('\n\n[CONTEXT]')[0]
    
    expected_question = "What is the main difference between fiber optic cables and twisted pair cables in terms of data transmission?"
    assert question == expected_question, f"Question mismatch: {question}"
    
    print("[PASS] Question loading consistent with 4A/4C")
    return question

def test_context_with_metadata():
    """Test context retrieval with metadata"""
    print("Testing context retrieval with metadata...")
    
    indexer = SemanticIndexer()
    indexer.load_index()
    
    question = "What is TCP protocol?"
    context, metadata = retrieve_context_with_metadata(question, indexer, k=3)
    
    assert len(context) > 0, "Context should not be empty"
    assert len(metadata) == 3, "Should return 3 metadata entries"
    assert all('chunk_id' in chunk for chunk in metadata), "All chunks should have chunk_id"
    assert all('source' in chunk for chunk in metadata), "All chunks should have source"
    assert all('score' in chunk for chunk in metadata), "All chunks should have score"
    
    print("[PASS] Context retrieval with metadata working")
    return context, metadata

def test_baseline_generation():
    """Test baseline answer generation"""
    print("Testing baseline answer generation...")
    
    test_input = """[QUESTION]
What is HTTP?

[CONTEXT]
HTTP (HyperText Transfer Protocol) is an application-layer protocol for distributed, collaborative, hypermedia information systems.

[INSTRUCTION]
Answer the question academically using only the given context."""
    
    answer = generate_baseline_answer(test_input)
    
    assert len(answer) > 0, "Baseline answer should not be empty"
    assert isinstance(answer, str), "Answer should be a string"
    
    print("[PASS] Baseline answer generation working")
    print(f"Sample baseline: {answer[:50]}...")
    return answer

def test_grounded_generation():
    """Test grounded answer generation"""
    print("Testing grounded answer generation...")
    
    test_input = """[QUESTION]
What is HTTP?

[CONTEXT]
HTTP (HyperText Transfer Protocol) is an application-layer protocol for distributed, collaborative, hypermedia information systems.

[INSTRUCTION]
Answer the question academically using only the given context."""
    
    answer = generate_grounded_answer(test_input)
    
    assert len(answer) > 0, "Grounded answer should not be empty"
    assert isinstance(answer, str), "Answer should be a string"
    
    print("[PASS] Grounded answer generation working")
    print(f"Sample grounded: {answer[:50]}...")
    return answer

def test_comparison_display():
    """Test comparison display functionality"""
    print("Testing comparison display...")
    
    question = "What is a network protocol?"
    baseline_answer = "A protocol is a set of rules."
    grounded_answer = "A network protocol defines the format and order of messages exchanged between communicating entities."
    
    metadata = [
        {'chunk_id': 'chunk_123', 'source': 'textbook', 'score': 0.85},
        {'chunk_id': 'chunk_456', 'source': 'textbook', 'score': 0.72}
    ]
    
    # This should print the comparison without errors
    print("\n" + "="*50)
    print("SAMPLE COMPARISON OUTPUT:")
    print("="*50)
    display_comparison(question, baseline_answer, grounded_answer, metadata)
    
    print("[PASS] Comparison display working")
    return True

def test_full_pipeline():
    """Test the complete comparison pipeline"""
    print("Testing full comparison pipeline...")
    
    # Load question
    sample = load_sample_question()
    question = sample['input'].split('[QUESTION]\n')[1].split('\n\n[CONTEXT]')[0]
    
    # Load indexer and get context
    indexer = SemanticIndexer()
    indexer.load_index()
    context, metadata = retrieve_context_with_metadata(question, indexer, k=2)
    
    # Create MCP input
    mcp_input = create_mcp_input(question, context)
    
    # Generate both answers
    baseline_answer = generate_baseline_answer(mcp_input)
    grounded_answer = generate_grounded_answer(mcp_input)
    
    # Verify outputs
    assert len(baseline_answer) > 0, "Baseline answer should not be empty"
    assert len(grounded_answer) > 0, "Grounded answer should not be empty"
    assert len(metadata) == 2, "Should have 2 metadata entries"
    
    print("[PASS] Full pipeline working")
    return baseline_answer, grounded_answer, metadata

def run_all_tests():
    """Run all Module 4D validation tests"""
    print("=" * 50)
    print("MODULE 4D VALIDATION TESTS")
    print("=" * 50)
    
    try:
        # Test 1: Question loading
        question = test_question_loading()
        
        # Test 2: Context with metadata
        context, metadata = test_context_with_metadata()
        
        # Test 3: Baseline generation
        baseline = test_baseline_generation()
        
        # Test 4: Grounded generation
        grounded = test_grounded_generation()
        
        # Test 5: Comparison display
        display_test = test_comparison_display()
        
        # Test 6: Full pipeline
        bl_answer, gr_answer, meta = test_full_pipeline()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED [SUCCESS]")
        print("Module 4D comparison interface working correctly!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        print("=" * 50)
        raise

if __name__ == "__main__":
    run_all_tests()