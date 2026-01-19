import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from modules.module5a_question_validation.question_validator import QuestionValidator

def test_questions():
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

if __name__ == "__main__":
    test_questions()