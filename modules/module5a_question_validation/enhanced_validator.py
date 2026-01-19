"""
Enhanced Module 5A: Advanced Question Validation Pipeline
Features: Weighted scoring, structured feedback, embedding-based unit matching
"""

import re
import json
import spacy
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Tuple, Optional

class EnhancedQuestionValidator:
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
        self.validation_weights = self.config['validation_weights']
        self.scoring_thresholds = self.config['scoring_thresholds']
        self.syllabus_units = self.config['syllabus_units']
        self.protocol_layers = self.config['protocol_layers']
        self.question_words = self.config['question_words']
        
        # Pre-compute unit embeddings for faster matching
        self._precompute_unit_embeddings()
        
        # Flatten keywords for backward compatibility
        self.all_keywords = []
        for unit_data in self.syllabus_units.values():
            self.all_keywords.extend(unit_data['keywords'])
    
    def _precompute_unit_embeddings(self):
        """Pre-compute embeddings for syllabus units"""
        self.unit_embeddings = {}
        for unit_id, unit_data in self.syllabus_units.items():
            # Combine title, description, and keywords for rich embedding
            unit_text = f"{unit_data['title']}. {unit_data['description']}. Keywords: {', '.join(unit_data['keywords'])}"
            embedding = self.sentence_model.encode([unit_text])[0]
            self.unit_embeddings[unit_id] = {
                'embedding': embedding,
                'title': unit_data['title'],
                'description': unit_data['description']
            }
    
    def correct_grammar_spelling(self, question: str) -> Tuple[str, List[str]]:
        """Step 1: Grammar and spelling correction with change tracking"""
        corrections_applied = []
        corrected = question
        
        # Apply spell corrections
        for typo, correct in self.spell_corrections.items():
            if re.search(r'\b' + typo + r'\b', corrected, flags=re.IGNORECASE):
                corrected = re.sub(r'\b' + typo + r'\b', correct, corrected, flags=re.IGNORECASE)
                corrections_applied.append(f"'{typo}' → '{correct}'")
        
        # Basic grammar fixes
        corrected = re.sub(r'\s+', ' ', corrected).strip()
        
        # Add question mark if missing
        if not corrected.endswith('?'):
            corrected += '?'
            corrections_applied.append("Added question mark")
            
        return corrected, corrections_applied
    
    def check_semantic_sanity(self, question: str) -> Tuple[float, Dict]:
        """Step 2: Semantic sanity check with scoring"""
        doc = self.nlp(question)
        
        # Check components
        has_verb = any(token.pos_ == "VERB" for token in doc)
        has_noun = any(token.pos_ in ["NOUN", "PROPN"] for token in doc)
        has_question_word = any(token.text.lower() in self.question_words for token in doc)
        word_count = len([token for token in doc if token.is_alpha])
        
        score = 0.0
        issues = []
        
        # Word count check (0.3 weight)
        if word_count >= self.scoring_thresholds['min_words']:
            score += 0.3
        else:
            issues.append(f"Too short ({word_count} words, minimum {self.scoring_thresholds['min_words']})")
        
        # Structure check (0.4 weight)
        if has_verb or has_question_word:
            score += 0.4
        else:
            issues.append("Missing verbs or question words")
        
        # Noun check (0.2 weight)
        if has_noun:
            score += 0.2
        else:
            issues.append("No nouns detected")
        
        # Gibberish check (0.1 weight)
        alpha_words = [token.text for token in doc if token.is_alpha]
        if alpha_words:
            short_words = [word for word in alpha_words if len(word) <= 2]
            if len(short_words) <= len(alpha_words) * self.scoring_thresholds['gibberish_threshold']:
                score += 0.1
            else:
                issues.append("Too many very short words (possible gibberish)")
        
        return score, {
            'score': score,
            'issues': issues,
            'word_count': word_count,
            'has_structure': has_verb or has_question_word
        }
    
    def check_rejection_rules(self, question: str) -> Tuple[bool, Dict]:
        """Check if question should be rejected based on new rules"""
        question_lower = question.lower()
        rejection_info = {'should_reject': False, 'reasons': []}
        
        # Check impossible scenarios
        if 'impossible_scenarios' in self.config['rejection_rules']:
            for rule in self.config['rejection_rules']['impossible_scenarios']:
                if all(keyword in question_lower for keyword in rule['keywords']):
                    rejection_info['should_reject'] = True
                    rejection_info['reasons'].append({
                        'type': 'impossible_scenario',
                        'message': rule['message'],
                        'reason': rule['reason']
                    })
        
        # Check ambiguous questions
        if 'ambiguous_questions' in self.config['rejection_rules']:
            for rule in self.config['rejection_rules']['ambiguous_questions']:
                for pattern in rule['patterns']:
                    if pattern in question_lower:
                        rejection_info['should_reject'] = True
                        rejection_info['reasons'].append({
                            'type': 'ambiguous',
                            'message': rule['message'],
                            'suggestion': rule['suggestion']
                        })
        
        # Check non-academic tone
        if 'non_academic_tone' in self.config['rejection_rules']:
            for rule in self.config['rejection_rules']['non_academic_tone']:
                if any(keyword in question_lower for keyword in rule['keywords']):
                    rejection_info['should_reject'] = True
                    rejection_info['reasons'].append({
                        'type': 'non_academic',
                        'message': rule['message'],
                        'suggestion': rule['suggestion']
                    })
        
        # Check out of syllabus
        if 'out_of_syllabus' in self.config['rejection_rules']:
            for rule in self.config['rejection_rules']['out_of_syllabus']:
                if any(keyword in question_lower for keyword in rule['keywords']):
                    rejection_info['should_reject'] = True
                    rejection_info['reasons'].append({
                        'type': 'out_of_syllabus',
                        'message': rule['message'],
                        'suggestion': rule['suggestion']
                    })
        
        return rejection_info['should_reject'], rejection_info
    
    def check_technical_accuracy(self, question: str) -> Tuple[float, Dict]:
        """Step 3: Technical accuracy with structured feedback"""
        question_lower = question.lower()
        score = 1.0  # Start with perfect score
        violations = []
        
        # Check protocol-layer mismatches
        for protocol, correct_layer in self.protocol_layers.items():
            if protocol in question_lower:
                for layer in ["physical layer", "data link layer", "network layer", "transport layer", "application layer"]:
                    if layer in question_lower and layer != correct_layer:
                        score = 0.0
                        violations.append({
                            'type': 'layer_mismatch',
                            'protocol': protocol,
                            'incorrect_layer': layer,
                            'correct_layer': correct_layer,
                            'severity': 'high',
                            'message': f"{protocol.upper()} operates at {correct_layer}, not {layer}"
                        })
        
        # Check misconceptions from config
        for rule in self.config['validation_rules']['misconceptions']:
            if all(keyword in question_lower for keyword in rule['keywords']):
                score = 0.0
                violations.append({
                    'type': 'misconception',
                    'message': rule['message'],
                    'severity': rule['severity'],
                    'suggestion': rule['suggestion']
                })
        
        # Check cross-layer errors from config
        for rule in self.config['validation_rules']['cross_layer_errors']:
            if all(keyword in question_lower for keyword in rule['keywords']):
                score = 0.0
                violations.append({
                    'type': 'cross_layer_error',
                    'message': rule['message'],
                    'severity': rule['severity'],
                    'suggestion': rule['suggestion']
                })
        
        return score, {
            'score': score,
            'violations': violations,
            'is_technically_sound': score > 0
        }
        """Step 3: Technical accuracy with structured feedback"""
        question_lower = question.lower()
        score = 1.0  # Start with perfect score
        violations = []
        
        # Check protocol-layer mismatches
        for protocol, correct_layer in self.protocol_layers.items():
            if protocol in question_lower:
                for layer in ["physical layer", "data link layer", "network layer", "transport layer", "application layer"]:
                    if layer in question_lower and layer != correct_layer:
                        score = 0.0
                        violations.append({
                            'type': 'layer_mismatch',
                            'protocol': protocol,
                            'incorrect_layer': layer,
                            'correct_layer': correct_layer,
                            'severity': 'high'
                        })
        
        # Check misconceptions from config
        for rule in self.config['validation_rules']['misconceptions']:
            if all(keyword in question_lower for keyword in rule['keywords']):
                score = 0.0
                violations.append({
                    'type': 'misconception',
                    'message': rule['message'],
                    'severity': rule['severity'],
                    'suggestion': rule['suggestion']
                })
        
        # Check cross-layer errors from config
        for rule in self.config['validation_rules']['cross_layer_errors']:
            if all(keyword in question_lower for keyword in rule['keywords']):
                score = 0.0
                violations.append({
                    'type': 'cross_layer_error',
                    'message': rule['message'],
                    'severity': rule['severity'],
                    'suggestion': rule['suggestion']
                })
        
        return score, {
            'score': score,
            'violations': violations,
            'is_technically_sound': score > 0
        }
    
    def check_syllabus_relevance_embedding(self, question: str) -> Tuple[float, Dict]:
        """Step 4: Embedding-based syllabus unit matching"""
        question_embedding = self.sentence_model.encode([question])[0]
        
        # Calculate similarities with all units
        unit_similarities = {}
        for unit_id, unit_data in self.unit_embeddings.items():
            similarity = np.dot(question_embedding, unit_data['embedding'])
            unit_similarities[unit_id] = {
                'similarity': float(similarity),
                'title': unit_data['title'],
                'description': unit_data['description']
            }
        
        # Find best match
        best_unit = max(unit_similarities.keys(), key=lambda x: unit_similarities[x]['similarity'])
        best_similarity = unit_similarities[best_unit]['similarity']
        
        # Calculate score based on similarity
        if best_similarity >= self.scoring_thresholds['similarity_threshold']:
            score = min(best_similarity * 2, 1.0)  # Scale similarity to 0-1
        else:
            score = 0.0
        
        return score, {
            'score': score,
            'best_unit': best_unit,
            'best_similarity': best_similarity,
            'unit_title': unit_similarities[best_unit]['title'],
            'all_similarities': unit_similarities,
            'is_relevant': score > 0
        }
    
    def validate_question(self, raw_question: str) -> Dict:
        """Enhanced validation pipeline with weighted scoring"""
        # Step 1: Grammar correction
        corrected_question, corrections = self.correct_grammar_spelling(raw_question)
        
        # Step 1.5: Check rejection rules first
        should_reject, rejection_info = self.check_rejection_rules(corrected_question)
        if should_reject:
            return {
                'original_question': raw_question,
                'corrected_question': corrected_question,
                'corrections_applied': corrections,
                'is_valid': False,
                'status': 'rejected',
                'final_score': 0.0,
                'rejection_reasons': rejection_info['reasons'],
                'feedback': {
                    'summary': 'Question rejected due to validation rules',
                    'issues': [reason['message'] for reason in rejection_info['reasons']],
                    'suggestions': [reason.get('suggestion', '') for reason in rejection_info['reasons'] if reason.get('suggestion')]
                }
            }
        
        # Step 2: Semantic sanity
        semantic_score, semantic_details = self.check_semantic_sanity(corrected_question)
        
        # Step 3: Technical accuracy
        technical_score, technical_details = self.check_technical_accuracy(corrected_question)
        
        # Step 4: Syllabus relevance
        relevance_score, relevance_details = self.check_syllabus_relevance_embedding(corrected_question)
        
        # Calculate weighted final score
        final_score = (
            semantic_score * self.validation_weights['semantic_sanity'] +
            technical_score * self.validation_weights['technical_accuracy'] +
            relevance_score * self.validation_weights['syllabus_relevance']
        )
        
        # Determine validation result
        if final_score >= self.scoring_thresholds['pass_threshold']:
            is_valid = True
            status = "valid"
        elif final_score >= self.scoring_thresholds['warning_threshold']:
            is_valid = False
            status = "warning"
        else:
            is_valid = False
            status = "rejected"
        
        # Generate structured feedback
        feedback = self._generate_structured_feedback(
            semantic_details, technical_details, relevance_details, final_score, status
        )
        
        return {
            'original_question': raw_question,
            'corrected_question': corrected_question,
            'corrections_applied': corrections,
            'is_valid': is_valid,
            'status': status,
            'final_score': round(final_score, 3),
            'component_scores': {
                'semantic_sanity': round(semantic_score, 3),
                'technical_accuracy': round(technical_score, 3),
                'syllabus_relevance': round(relevance_score, 3)
            },
            'matched_unit': relevance_details['best_unit'] if relevance_details['is_relevant'] else None,
            'unit_title': relevance_details['unit_title'] if relevance_details['is_relevant'] else None,
            'feedback': feedback,
            'detailed_analysis': {
                'semantic': semantic_details,
                'technical': technical_details,
                'relevance': relevance_details
            }
        }
    
    def _generate_structured_feedback(self, semantic_details, technical_details, relevance_details, final_score, status):
        """Generate human-readable structured feedback"""
        feedback = {
            'summary': '',
            'issues': [],
            'suggestions': [],
            'strengths': []
        }
        
        # Generate summary based on status
        if status == "valid":
            feedback['summary'] = f"Question validated successfully (Score: {final_score:.2f})"
            if relevance_details['is_relevant']:
                feedback['summary'] += f" - Matches {relevance_details['unit_title']}"
        elif status == "warning":
            feedback['summary'] = f"Question has some issues but may be acceptable (Score: {final_score:.2f})"
        else:
            feedback['summary'] = f"Question rejected due to significant issues (Score: {final_score:.2f})"
        
        # Add semantic issues
        if semantic_details['issues']:
            feedback['issues'].extend([f"Semantic: {issue}" for issue in semantic_details['issues']])
        
        # Add technical violations
        if technical_details['violations']:
            for violation in technical_details['violations']:
                message = violation.get('message', 'Technical accuracy issue')
                feedback['issues'].append(f"Technical: {message}")
                if 'suggestion' in violation:
                    feedback['suggestions'].append(violation['suggestion'])
        
        # Add relevance feedback
        if not relevance_details['is_relevant']:
            feedback['issues'].append("Low relevance to Computer Networks syllabus")
            feedback['suggestions'].append("Try including networking concepts, protocols, or layer-specific terms")
        
        # Add strengths
        if semantic_details['has_structure']:
            feedback['strengths'].append("Well-structured question format")
        if technical_details['is_technically_sound']:
            feedback['strengths'].append("Technically accurate concepts")
        if relevance_details['is_relevant']:
            feedback['strengths'].append(f"Relevant to {relevance_details['unit_title']}")
        
        return feedback

def test_enhanced_validator():
    """Test the enhanced validation system"""
    validator = EnhancedQuestionValidator()
    
    test_questions = [
        "What is TCP?",
        "HTTP operates at transport layer",
        "wht is tcps protocl",
        "How does routing work?",
        "What is machine learning?",
        "asdf qwerty"
    ]
    
    print("=== Enhanced Question Validation Results ===\\n")
    
    for question in test_questions:
        result = validator.validate_question(question)
        
        print(f"Question: '{question}'")
        print(f"Status: {result['status'].upper()} (Score: {result['final_score']})")
        print(f"Corrected: '{result['corrected_question']}'")
        if result['matched_unit']:
            print(f"Unit: {result['unit_title']}")
        print(f"Summary: {result['feedback']['summary']}")
        
        if result['feedback']['issues']:
            print("Issues:", "; ".join(result['feedback']['issues']))
        if result['feedback']['suggestions']:
            print("Suggestions:", "; ".join(result['feedback']['suggestions']))
        
        print("-" * 60)

if __name__ == "__main__":
    test_enhanced_validator()