"""
Module 5A: Question Validator Testing
Interactive testing for both basic and enhanced validators
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.module5a_question_validation.question_validator import QuestionValidator
from modules.module5a_question_validation.enhanced_validator import EnhancedQuestionValidator

def test_basic_validator():
    """Test basic question validator"""
    print("=== Basic Question Validator ===")
    validator = QuestionValidator()
    
    while True:
        question = input("\nEnter question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        result = validator.validate_question(question)
        
        print(f"Original: '{result['original_question']}'")
        print(f"Corrected: '{result['corrected_question']}'")
        print(f"Valid: {result['is_valid']}")
        print(f"Message: {result['final_message']}")
        print("-" * 50)

def test_enhanced_validator():
    """Test enhanced question validator"""
    print("=== Enhanced Question Validator ===")
    print("Features: Weighted scoring, structured feedback, unit matching")
    
    validator = EnhancedQuestionValidator()
    
    while True:
        question = input("\nEnter question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        result = validator.validate_question(question)
        
        print(f"\n📝 Original: '{result['original_question']}'")
        print(f"✏️  Corrected: '{result['corrected_question']}'")
        
        if result['corrections_applied']:
            print(f"🔧 Corrections: {', '.join(result['corrections_applied'])}")
        
        # Status with color coding
        status_emoji = {"valid": "✅", "warning": "⚠️", "rejected": "❌"}
        print(f"{status_emoji.get(result['status'], '❓')} Status: {result['status'].upper()} (Score: {result['final_score']})")
        
        # Component scores
        if 'component_scores' in result:
            scores = result['component_scores']
            print(f"📊 Scores: Semantic({scores['semantic_sanity']}) + Technical({scores['technical_accuracy']}) + Relevance({scores['syllabus_relevance']})")
        
        # Unit matching
        if result.get('matched_unit'):
            print(f"📚 Unit: {result['unit_title']}")
        
        # Rejection reasons
        if 'rejection_reasons' in result:
            print(f"🚫 Rejection reasons:")
            for reason in result['rejection_reasons']:
                print(f"   - {reason['message']}")
        
        # Feedback
        if 'feedback' in result:
            feedback = result['feedback']
            print(f"💬 {feedback['summary']}")
            
            if feedback.get('issues'):
                print(f"⚠️  Issues: {'; '.join(feedback['issues'])}")
            
            if feedback.get('suggestions'):
                print(f"💡 Suggestions: {'; '.join([s for s in feedback['suggestions'] if s])}")
            
            if feedback.get('strengths'):
                print(f"👍 Strengths: {'; '.join(feedback['strengths'])}")
        
        print("-" * 60)

def main():
    """Main testing interface"""
    print("🎓 Module 5A: Question Validator Testing")
    print("Choose validator to test:")
    print("1. Basic Validator")
    print("2. Enhanced Validator (Recommended)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_basic_validator()
    elif choice == "2":
        test_enhanced_validator()
    else:
        print("Invalid choice. Using Enhanced Validator...")
        test_enhanced_validator()

if __name__ == "__main__":
    main()