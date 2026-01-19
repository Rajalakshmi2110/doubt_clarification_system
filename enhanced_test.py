import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from modules.module5a_question_validation.enhanced_validator import EnhancedQuestionValidator

def interactive_test():
    validator = EnhancedQuestionValidator()
    
    print("=== Enhanced Question Validator ===")
    print("Features: Weighted scoring, structured feedback, unit matching")
    print()
    
    while True:
        question = input("Enter question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        result = validator.validate_question(question)
        
        print(f"\\n📝 Original: '{result['original_question']}'")
        print(f"✏️  Corrected: '{result['corrected_question']}'")
        
        if result['corrections_applied']:
            print(f"🔧 Corrections: {', '.join(result['corrections_applied'])}")
        
        # Status with color coding
        status_emoji = {"valid": "✅", "warning": "⚠️", "rejected": "❌"}
        print(f"{status_emoji.get(result['status'], '❓')} Status: {result['status'].upper()} (Score: {result['final_score']})")
        
        # Component scores (only for non-rejected questions)
        if 'component_scores' in result:
            scores = result['component_scores']
            print(f"📊 Scores: Semantic({scores['semantic_sanity']}) + Technical({scores['technical_accuracy']}) + Relevance({scores['syllabus_relevance']})")
        
        # Unit matching
        if result.get('matched_unit'):
            print(f"📚 Unit: {result['unit_title']}")
        
        # Show rejection reasons if present
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

if __name__ == "__main__":
    interactive_test()