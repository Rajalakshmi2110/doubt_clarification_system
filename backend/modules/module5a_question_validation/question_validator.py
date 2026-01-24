"""
Enhanced Module 5A: Intelligent Question Validation Pipeline
ML-based approach that learns from your dataset automatically
"""

import re
import json
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import FLAN-T5 checker
try:
    from .flan_t5_validator import FLANT5RelevanceChecker
    FLAN_T5_AVAILABLE = True
except ImportError:
    FLAN_T5_AVAILABLE = False
    print("FLAN-T5 not available, using embedding-based validation")

class EnhancedQuestionValidator:
    def __init__(self):
        # Load models
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize FLAN-T5 if available
        if FLAN_T5_AVAILABLE:
            try:
                self.flan_t5_checker = FLANT5RelevanceChecker()
                self.use_flan_t5 = True
                print("Using FLAN-T5 for contextual relevance checking")
            except Exception as e:
                print(f"FLAN-T5 initialization failed: {e}")
                self.use_flan_t5 = False
        else:
            self.use_flan_t5 = False
        
        # Fallback: Learn from networking dataset
        self.networking_embeddings = self._learn_from_dataset()
        self.outlier_detector = self._train_outlier_detector()
        
        print("Intelligent validator loaded - learned from your networking questions")
    
    def _learn_from_dataset(self):
        """Learn what networking questions look like from your dataset"""
        questions = []
        
        try:
            with open('../data/dataset/questions_collection.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        question = line.split('.', 1)[1].strip() if '.' in line else line.strip()
                        questions.append(question)
        except FileNotFoundError:
            # Fallback to basic networking terms if file not found
            questions = [
                "What is TCP?", "How does routing work?", "Explain OSI model",
                "What is HTTP?", "How does DNS work?", "What is Ethernet?"
            ]
        
        embeddings = self.sentence_model.encode(questions)
        return embeddings
    
    def _train_outlier_detector(self):
        """Train outlier detection for non-networking questions"""
        detector = IsolationForest(contamination=0.1, random_state=42)
        detector.fit(self.networking_embeddings)
        return detector
    
    def correct_grammar_spelling(self, question: str) -> Tuple[str, List[str]]:
        """Basic grammar and spelling correction"""
        corrections_applied = []
        corrected = question.strip()
        
        # Basic spell corrections
        simple_fixes = {
            'wht': 'what', 'hw': 'how', 'wat': 'what',
            'protocl': 'protocol', 'netwrk': 'network', 'explian': 'explain'
        }
        
        for typo, correct in simple_fixes.items():
            if re.search(r'\b' + typo + r'\b', corrected, flags=re.IGNORECASE):
                corrected = re.sub(r'\b' + typo + r'\b', correct, corrected, flags=re.IGNORECASE)
                corrections_applied.append(f"'{typo}' → '{correct}'")
        
        # Add question mark if missing
        if not corrected.endswith('?'):
            corrected += '?'
            corrections_applied.append("Added question mark")
            
        return corrected, corrections_applied
    
    def check_semantic_sanity(self, question: str) -> Tuple[float, Dict]:
        """Check if question makes semantic sense"""
        doc = self.nlp(question)
        
        has_verb = any(token.pos_ == "VERB" for token in doc)
        has_noun = any(token.pos_ in ["NOUN", "PROPN"] for token in doc)
        has_question_word = any(token.text.lower() in ['what', 'how', 'why', 'when', 'where', 'which', 'explain', 'describe'] for token in doc)
        word_count = len([token for token in doc if token.is_alpha])
        
        score = 0.0
        issues = []
        
        if word_count >= 3:
            score += 0.4
        else:
            issues.append(f"Too short ({word_count} words)")
        
        if has_verb or has_question_word:
            score += 0.4
        else:
            issues.append("Missing question structure")
        
        if has_noun:
            score += 0.2
        else:
            issues.append("No nouns detected")
        
        return score, {
            'score': score,
            'issues': issues,
            'word_count': word_count,
            'has_structure': has_verb or has_question_word
        }
    
    def check_networking_relevance(self, question: str) -> Tuple[float, Dict]:
        """ML-based networking relevance detection with FLAN-T5 or embeddings"""
        
        # Use FLAN-T5 if available
        if self.use_flan_t5:
            return self.flan_t5_checker.check_relevance_with_flan_t5(question)
        
        # Fallback to embedding-based approach
        return self._check_relevance_with_embeddings(question)
    
    def _check_relevance_with_embeddings(self, question: str) -> Tuple[float, Dict]:
        """Embedding-based relevance detection (fallback method)"""
        question_embedding = self.sentence_model.encode([question])
        
        # Calculate similarity to known networking questions
        similarities = cosine_similarity(question_embedding, self.networking_embeddings)[0]
        max_similarity = np.max(similarities)
        avg_similarity = np.mean(similarities)
        
        # Stricter thresholds for networking relevance
        if max_similarity < 0.3:  # Very low similarity to any networking question
            return 0.0, {
                'score': 0.0,
                'is_relevant': False,
                'max_similarity': float(max_similarity),
                'avg_similarity': float(avg_similarity),
                'reason': f'Too low similarity to networking topics (max: {max_similarity:.3f})',
                'method': 'Embedding similarity'
            }
        
        # Check for explicit non-networking terms and contexts
        non_networking_terms = ['python', 'java', 'programming', 'fear', 'emotion', 'psychology', 'cooking', 'sports', 'linkedin', 'facebook', 'twitter', 'social', 'career', 'job', 'professional']
        social_networking_patterns = ['networking in', 'networking on', 'professional networking', 'business networking', 'career networking']
        
        question_lower = question.lower()
        
        # Check for social/professional networking context
        for pattern in social_networking_patterns:
            if pattern in question_lower:
                return 0.0, {
                    'score': 0.0,
                    'is_relevant': False,
                    'max_similarity': float(max_similarity),
                    'avg_similarity': float(avg_similarity),
                    'reason': f'Detected social/professional networking context: "{pattern}"',
                    'method': 'Pattern matching'
                }
        
        # Check for other non-networking terms
        for term in non_networking_terms:
            if term in question_lower:
                return 0.0, {
                    'score': 0.0,
                    'is_relevant': False,
                    'max_similarity': float(max_similarity),
                    'avg_similarity': float(avg_similarity),
                    'reason': f'Contains non-networking term: "{term}"',
                    'method': 'Term filtering'
                }
        
        # Calculate relevance score with stricter scaling
        relevance_score = min(max_similarity * 1.5, 1.0)  # More conservative scoring
        
        return relevance_score, {
            'score': relevance_score,
            'is_relevant': relevance_score > 0.4,
            'max_similarity': float(max_similarity),
            'avg_similarity': float(avg_similarity),
            'reason': f'ML similarity analysis (max: {max_similarity:.3f})',
            'method': 'Embedding similarity'
        }
    
    def validate_question(self, raw_question: str) -> Dict:
        """Intelligent validation pipeline using ML"""
        # Step 1: Grammar correction
        corrected_question, corrections = self.correct_grammar_spelling(raw_question)
        
        # Step 2: Semantic sanity
        semantic_score, semantic_details = self.check_semantic_sanity(corrected_question)
        
        # Step 3: ML-based networking relevance
        relevance_score, relevance_details = self.check_networking_relevance(corrected_question)
        
        # Calculate weighted final score - prioritize networking relevance
        final_score = (semantic_score * 0.2 + relevance_score * 0.8)
        
        # Stricter validation thresholds
        if relevance_score >= 0.5 and final_score >= 0.6:
            is_valid = True
            status = "VALID"
        elif relevance_score >= 0.3 and final_score >= 0.4:
            is_valid = False
            status = "WARNING"
        else:
            is_valid = False
            status = "REJECTED"
        
        # Generate dynamic feedback based on actual analysis
        issues = []
        suggestions = []
        strengths = []
        
        # Add semantic issues if any
        if semantic_details['issues']:
            issues.extend(semantic_details['issues'])
        
        # Dynamic relevance feedback
        if not relevance_details.get('is_relevant', False):
            if relevance_details.get('method') == 'Term filtering (pre-filter)':
                issues.append(f"Contains non-networking terminology")
                suggestions.append("Focus on computer networking concepts like protocols, layers, or network devices")
            elif relevance_details.get('method') == 'Pattern matching (pre-filter)':
                issues.append("Question appears to be about social/professional networking rather than computer networking")
                suggestions.append("Ask about technical networking topics like TCP, routing, or network protocols")
            elif relevance_score < 0.3:
                issues.append("Very low similarity to Computer Networks syllabus topics")
                suggestions.append("Include specific networking terms like TCP, IP, HTTP, routing, or OSI layers")
            else:
                issues.append("Moderate relevance to Computer Networks syllabus")
                suggestions.append("Try to be more specific about networking protocols or concepts")
        
        # Dynamic strengths based on actual question analysis
        if semantic_details['has_structure']:
            strengths.append("Well-structured question format")
        
        if relevance_details.get('is_relevant', False):
            if relevance_score > 0.8:
                strengths.append("Highly relevant to networking domain")
            elif relevance_score > 0.6:
                strengths.append("Good relevance to networking concepts")
            else:
                strengths.append("Some relevance to networking domain")
        
        # Add specific strengths based on question content
        question_lower = corrected_question.lower()
        networking_terms = ['tcp', 'udp', 'http', 'ip', 'osi', 'routing', 'ethernet', 'dns', 'ftp', 'smtp', 'handshake', 'protocol', 'network']
        found_terms = [term for term in networking_terms if term in question_lower]
        if found_terms:
            strengths.append(f"Contains specific networking terms: {', '.join(found_terms)}")
            # If question contains networking terms but was rejected, override to WARNING
            if status == "REJECTED" and len(found_terms) > 0:
                status = "WARNING"
                final_score = max(final_score, 0.5)
                is_valid = False
        
        # Dynamic suggestions based on question type
        if semantic_details['word_count'] < 3:
            suggestions.append("Consider adding more descriptive words to clarify your question")
        
        if not semantic_details['has_structure']:
            suggestions.append("Rephrase as a clear question using words like 'what', 'how', or 'explain'")
        
        feedback = {
            'summary': f"Question {status} (Score: {final_score:.2f})",
            'issues': issues,
            'suggestions': suggestions,
            'strengths': strengths
        }
        
        return {
            'original_question': raw_question,
            'corrected_question': corrected_question,
            'corrections_applied': corrections,
            'is_valid': is_valid,
            'status': status,
            'final_score': round(final_score, 3),
            'validation_method': 'FLAN-T5' if self.use_flan_t5 else 'Embedding-based',
            'component_scores': {
                'semantic_sanity': round(semantic_score, 3),
                'syllabus_relevance': round(relevance_score, 3)
            },
            'relevance_details': relevance_details,
            'feedback': feedback
        }

def test_enhanced_validator():
    """Test the intelligent validation system"""
    validator = EnhancedQuestionValidator()
    
    test_questions = [
        "What is TCP?",
        "What is set in python?",
        "How does HTTP work?",
        "What is machine learning?",
        "how to overcome fear",
        "networking in linkedin"
    ]
    
    print("\n=== Intelligent Question Validation Results ===\n")
    
    for question in test_questions:
        result = validator.validate_question(question)
        
        print(f"Question: '{question}'")
        print(f"Status: {result['status'].upper()} (Score: {result['final_score']})")
        print(f"Summary: {result['feedback']['summary']}")
        print("-" * 60)

if __name__ == "__main__":
    test_enhanced_validator()