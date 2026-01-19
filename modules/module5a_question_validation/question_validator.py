"""
Module 5A: Question Validation Pipeline
Purpose: Validate and correct user questions before processing
"""

import re
import json
import spacy
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Tuple

class QuestionValidator:
    def __init__(self):
        # Load models
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load configuration from JSON
        config_path = Path(__file__).parent / 'validation_config.json'
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Extract config data
        self.spell_corrections = self.config['spell_corrections']
        self.syllabus_keywords = self.config['syllabus_keywords']
        self.protocol_layers = self.config['protocol_layers']
        self.thresholds = self.config['thresholds']
        self.question_words = self.config['question_words']
        self.syllabus_topics = self.config['syllabus_topics']
        
        # Flatten keywords for relevance checking
        self.all_keywords = []
        for unit_keywords in self.syllabus_keywords.values():
            self.all_keywords.extend(unit_keywords)
    
    def correct_grammar_spelling(self, question: str) -> str:
        """Step 1: Basic grammar fixes (minimal spell correction)"""
        # Apply specific corrections from config
        corrected = question
        for typo, correct in self.spell_corrections.items():
            corrected = re.sub(r'\b' + typo + r'\b', correct, corrected, flags=re.IGNORECASE)
        
        # Basic grammar fixes
        corrected = re.sub(r'\s+', ' ', corrected)  # Multiple spaces
        corrected = corrected.strip()
        
        # Ensure question ends with ?
        if not corrected.endswith('?'):
            corrected += '?'
            
        return corrected
    
    def check_semantic_sanity(self, question: str) -> Tuple[bool, str]:
        """Step 2: Check if question is semantically meaningful"""
        doc = self.nlp(question)
        
        # Check for minimum requirements
        has_verb = any(token.pos_ == "VERB" for token in doc)
        has_noun = any(token.pos_ in ["NOUN", "PROPN"] for token in doc)
        has_question_word = any(token.text.lower() in self.question_words for token in doc)
        word_count = len([token for token in doc if token.is_alpha])
        
        # Sanity checks
        if word_count < self.thresholds['min_words']:
            return False, f"Question too short (minimum {self.thresholds['min_words']} words required)"
        
        # Check for basic question structure (verb OR question word)
        if not (has_verb or has_question_word or has_noun):
            return False, "Question lacks meaningful content (no verbs, question words, or nouns found)"
        
        # Check for excessive gibberish (all words are very short or non-alphabetic)
        alpha_words = [token.text for token in doc if token.is_alpha]
        if len(alpha_words) == 0:
            return False, "Question contains no valid words"
        
        # Check if most words are very short (likely gibberish)
        short_words = [word for word in alpha_words if len(word) <= 2]
        if len(short_words) > len(alpha_words) * self.thresholds['gibberish_threshold']:
            return False, "Question appears to be gibberish (too many very short words)"
        
        return True, "Question is semantically valid"
    
    def check_technical_accuracy(self, question: str) -> Tuple[bool, str]:
        """Step 4: Check for obvious technical inaccuracies"""
        question_lower = question.lower()
        
        # Check for protocol-layer mismatches
        for protocol, correct_layer in self.protocol_layers.items():
            if protocol in question_lower:
                # Check if question mentions wrong layer
                for layer in ["physical layer", "data link layer", "network layer", "transport layer", "application layer"]:
                    if layer in question_lower and layer != correct_layer:
                        return False, f"Technical error: {protocol.upper()} operates at {correct_layer}, not {layer}"
        
        # Check for common misconceptions from config
        for rule in self.config['validation_rules']['misconceptions']:
            if all(keyword in question_lower for keyword in rule['keywords']):
                return False, f"Technical error: {rule['message']}"
        
        # Check for cross-layer conceptual errors from config
        for rule in self.config['validation_rules']['cross_layer_errors']:
            if all(keyword in question_lower for keyword in rule['keywords']):
                return False, f"Conceptual error: {rule['message']}"
        
        return True, "No technical errors detected"
    
    def check_syllabus_relevance(self, question: str) -> Tuple[bool, str, float]:
        """Step 3: Check if question is relevant to Computer Networks syllabus"""
        question_lower = question.lower()
        
        # Direct keyword matching (exact words only)
        keyword_matches = []
        question_words = question_lower.split()
        
        for keyword in self.all_keywords:
            # Check for exact word matches or common variations
            if keyword in question_words or any(keyword in word for word in question_words if len(word) > 3):
                keyword_matches.append(keyword)
        
        # Calculate relevance score
        if keyword_matches:
            relevance_score = len(keyword_matches) / len(question.split()) * 100
            relevance_score = min(relevance_score, 100)  # Cap at 100%
            
            if relevance_score >= self.thresholds['relevance_threshold']:  # Configurable threshold
                return True, f"Relevant to Computer Networks (keywords: {', '.join(keyword_matches)})", relevance_score
        
        # Semantic similarity check with syllabus topics from config
        question_embedding = self.sentence_model.encode([question])
        topic_embeddings = self.sentence_model.encode(self.syllabus_topics)
        
        similarities = np.dot(question_embedding, topic_embeddings.T)[0]
        max_similarity = float(np.max(similarities))
        
        if max_similarity >= self.thresholds['similarity_threshold']:  # Configurable threshold
            best_topic = self.syllabus_topics[np.argmax(similarities)]
            return True, f"Semantically relevant to: {best_topic}", max_similarity * 100
        
        return False, "Question not relevant to Computer Networks syllabus", max_similarity * 100
    
    def validate_question(self, raw_question: str) -> Dict:
        """Complete validation pipeline"""
        result = {
            "original_question": raw_question,
            "corrected_question": "",
            "is_valid": False,
            "validation_steps": {},
            "final_message": ""
        }
        
        # Step 1: Grammar and spelling correction
        corrected_question = self.correct_grammar_spelling(raw_question)
        result["corrected_question"] = corrected_question
        result["validation_steps"]["grammar_correction"] = {
            "applied": corrected_question != raw_question,
            "message": "Grammar and spelling corrected" if corrected_question != raw_question else "No corrections needed"
        }
        
        # Step 2: Semantic sanity check
        is_sane, sanity_message = self.check_semantic_sanity(corrected_question)
        result["validation_steps"]["semantic_sanity"] = {
            "passed": is_sane,
            "message": sanity_message
        }
        
        if not is_sane:
            result["final_message"] = f"Invalid question: {sanity_message}"
            return result
        
        # Step 3: Technical accuracy check
        is_accurate, accuracy_message = self.check_technical_accuracy(corrected_question)
        result["validation_steps"]["technical_accuracy"] = {
            "passed": is_accurate,
            "message": accuracy_message
        }
        
        if not is_accurate:
            result["final_message"] = f"Technically incorrect: {accuracy_message}"
            return result
        
        # Step 4: Syllabus relevance check
        is_relevant, relevance_message, relevance_score = self.check_syllabus_relevance(corrected_question)
        result["validation_steps"]["syllabus_relevance"] = {
            "passed": is_relevant,
            "message": relevance_message,
            "score": round(relevance_score, 2)
        }
        
        if not is_relevant:
            result["final_message"] = f"Off-topic question: {relevance_message} (Score: {relevance_score:.1f}%)"
            return result
        
        # All validations passed
        result["is_valid"] = True
        result["final_message"] = f"Question validated successfully. {relevance_message}"
        
        return result

def test_question_validator():
    """Test the question validation pipeline"""
    validator = QuestionValidator()
    
    test_questions = [
        "What is TCP?",  # Valid
        "Explain the OSI model layers",  # Valid
        "How does routing work in networks?",  # Valid
        "What is machine learning?",  # Off-topic
        "asdf qwerty zxcv?",  # Nonsensical
        "TCP",  # Too short
        "Wht is TCP protocl",  # Spelling errors
    ]
    
    print("=== Question Validation Test Results ===\n")
    
    for question in test_questions:
        result = validator.validate_question(question)
        
        print(f"Original: '{result['original_question']}'")
        print(f"Corrected: '{result['corrected_question']}'")
        print(f"Valid: {result['is_valid']}")
        print(f"Message: {result['final_message']}")
        
        # Show validation steps
        for step, details in result['validation_steps'].items():
            print(f"  {step}: {details}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_question_validator()