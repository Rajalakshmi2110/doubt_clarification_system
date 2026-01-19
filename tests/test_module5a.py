"""
Test Module 5A: Question Validation Pipeline
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

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