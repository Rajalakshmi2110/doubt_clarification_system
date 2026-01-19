"""
Test Module 5A: Question Validation Pipeline
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.module5a_question_validation.question_validator import QuestionValidator

def test_validation_pipeline():
    """Test all validation components"""
    validator = QuestionValidator()
    
    # Test cases covering all scenarios
    test_cases = [
        # Valid Computer Networks questions
        ("What is TCP?", True, "Valid networking question"),
        ("Explain the OSI model layers", True, "Valid networking question"),
        ("How does routing work in networks?", True, "Valid networking question"),
        ("Define ethernet protocol", True, "Valid networking question"),
        
        # Grammar/spelling corrections
        ("Wht is TCP protocl", True, "Should auto-correct spelling"),
        ("explain osi model", True, "Should add question mark"),
        
        # Technical correctness tests
        ("HTTP operates at transport layer", False, "Technical error - HTTP is application layer"),
        ("TCP operates at network layer", False, "Technical error - TCP is transport layer"),
        ("ICMP operates at transport layer", False, "Technical error - ICMP is network layer"),
        ("What would happen if TCP used MAC addresses?", False, "Cross-layer conceptual error"),
        ("If DNS used TCP-like error correction?", True, "Valid hypothetical question"),
        ("Compare TCP and UDP at transport layer", True, "Technically correct comparison"),
        
        # Off-topic questions
        ("What is machine learning?", False, "Not Computer Networks related"),
        ("How to cook pasta?", False, "Completely off-topic"),
        
        # Nonsensical questions
        ("asdf qwerty zxcv?", False, "Gibberish text"),
        ("TCP", False, "Too short"),
        ("", False, "Empty question"),
        
        # Edge cases
        ("Is TCP reliable?", True, "Simple yes/no question"),
        ("Compare TCP and UDP protocols", True, "Comparison question"),
    ]
    
    print("=" * 60)
    print("MODULE 5A: QUESTION VALIDATION TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(test_cases)
    
    for i, (question, expected_valid, description) in enumerate(test_cases, 1):
        print(f"\nTest {i}: {description}")
        print(f"Input: '{question}'")
        
        result = validator.validate_question(question)
        
        print(f"Corrected: '{result['corrected_question']}'")
        print(f"Expected Valid: {expected_valid}, Actual Valid: {result['is_valid']}")
        print(f"Message: {result['final_message']}")
        
        # Check if test passed
        if result['is_valid'] == expected_valid:
            print("PASS")
            passed += 1
        else:
            print("FAIL")
        
        print("-" * 40)
    
    print(f"\nSUMMARY: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("All tests passed! Module 5A is working correctly.")
    else:
        print(f"Warning: {total-passed} tests failed. Review implementation.")

if __name__ == "__main__":
    test_validation_pipeline()