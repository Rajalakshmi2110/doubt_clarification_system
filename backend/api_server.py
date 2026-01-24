from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules', 'module5a_question_validation'))

try:
    from modules.module5a_question_validation.question_validator import EnhancedQuestionValidator
    validator = EnhancedQuestionValidator()
    print("Question validator loaded successfully")
except Exception as e:
    print(f"Error loading validator: {e}")
    validator = None

app = Flask(__name__)
CORS(app)

@app.route('/api/validate', methods=['POST'])
def validate_question():
    try:
        if not validator:
            return jsonify({'error': 'Validator not available'}), 500
        
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Question is required'}), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400
        
        result = validator.validate_question(question)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'validator_loaded': validator is not None
    })

if __name__ == '__main__':
    print("Starting Academic Doubt Clarification API Server...")
    print("Frontend: http://localhost:3000")
    print("Backend API: http://localhost:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)