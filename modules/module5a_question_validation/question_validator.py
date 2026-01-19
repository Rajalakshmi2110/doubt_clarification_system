"""
Module 5A: Question Validation Pipeline
Purpose: Validate and correct user questions before processing
"""

import re
import spacy
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Tuple

class QuestionValidator:
    def __init__(self):
        # Load models
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Computer Networks syllabus keywords with layer mapping
        self.syllabus_keywords = {
            "unit1": ["network", "protocol", "osi", "tcp/ip", "data communication", "transmission"],
            "unit2": ["physical layer", "data link", "ethernet", "frame", "error detection", "flow control"],
            "unit3": ["network layer", "routing", "ip", "subnet", "nat", "dhcp", "icmp"],
            "unit4": ["transport layer", "tcp", "udp", "socket", "port", "congestion control"],
            "unit5": ["application layer", "http", "dns", "ftp", "smtp", "snmp"]
        }
        
        # Protocol-to-layer mapping for technical validation
        self.protocol_layers = {
            "http": "application layer",
            "https": "application layer", 
            "ftp": "application layer",
            "smtp": "application layer",
            "dns": "application layer",
            "snmp": "application layer",
            "tcp": "transport layer",
            "udp": "transport layer",
            "ip": "network layer",
            "icmp": "network layer",
            "dhcp": "network layer",
            "ethernet": "data link layer",
            "arp": "data link layer"
        }
        
        # Flatten keywords for relevance checking
        self.all_keywords = []
        for unit_keywords in self.syllabus_keywords.values():
            self.all_keywords.extend(unit_keywords)
    
    def correct_grammar_spelling(self, question: str) -> str:
        """Step 1: Basic grammar fixes (minimal spell correction)"""
        # Simple typo corrections for common networking terms
        corrections = {
            'protocl': 'protocol',
            'protcol': 'protocol', 
            'netwrk': 'network',
            'routng': 'routing',
            'wht': 'what',
            'hw': 'how',
            'explian': 'explain',
            'defin': 'define',
            'tcps': 'tcp',
            'udps': 'udp',
            'https': 'http',
            'ftps': 'ftp',
            'smtps': 'smtp'
        }
        
        # Apply specific corrections
        corrected = question
        for typo, correct in corrections.items():
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
        has_question_word = any(token.text.lower() in ["what", "how", "why", "when", "where", "which", "who"] for token in doc)
        word_count = len([token for token in doc if token.is_alpha])
        
        # Sanity checks
        if word_count < 3:
            return False, "Question too short (minimum 3 words required)"
        
        # Check for basic question structure (verb OR question word)
        if not (has_verb or has_question_word or has_noun):
            return False, "Question lacks meaningful content (no verbs, question words, or nouns found)"
        
        # Check for excessive gibberish (all words are very short or non-alphabetic)
        alpha_words = [token.text for token in doc if token.is_alpha]
        if len(alpha_words) == 0:
            return False, "Question contains no valid words"
        
        # Check if most words are very short (likely gibberish)
        short_words = [word for word in alpha_words if len(word) <= 2]
        if len(short_words) > len(alpha_words) * 0.8:
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
        
        # Check for common misconceptions
        misconceptions = [
            (["http", "transport"], "HTTP operates at application layer, not transport layer"),
            (["tcp", "application"], "TCP operates at transport layer, not application layer"),
            (["ip", "transport"], "IP operates at network layer, not transport layer"),
            (["ethernet", "network"], "Ethernet operates at data link layer, not network layer")
        ]
        
        for keywords, error_msg in misconceptions:
            if all(keyword in question_lower for keyword in keywords):
                return False, f"Technical error: {error_msg}"
        
        # Check for cross-layer conceptual errors
        cross_layer_errors = [
            (["tcp", "mac"], "TCP operates at transport layer and doesn't handle MAC addresses (data link layer)"),
            (["http", "mac"], "HTTP operates at application layer and doesn't handle MAC addresses (data link layer)"),
            (["ip", "mac"], "IP addresses and MAC addresses serve different purposes at different layers"),
            (["tcp", "physical"], "TCP operates at transport layer, not physical layer"),
            (["http", "routing"], "HTTP operates at application layer and doesn't handle routing (network layer function)")
        ]
        
        for keywords, error_msg in cross_layer_errors:
            if all(keyword in question_lower for keyword in keywords):
                return False, f"Conceptual error: {error_msg}"
        
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
            
            if relevance_score >= 10:  # At least 10% keyword density
                return True, f"Relevant to Computer Networks (keywords: {', '.join(keyword_matches)})", relevance_score
        
        # Semantic similarity check with syllabus topics
        syllabus_topics = [
            "computer networks and protocols",
            "data link layer and ethernet",
            "network layer and routing",
            "transport layer tcp udp",
            "application layer protocols"
        ]
        
        question_embedding = self.sentence_model.encode([question])
        topic_embeddings = self.sentence_model.encode(syllabus_topics)
        
        similarities = np.dot(question_embedding, topic_embeddings.T)[0]
        max_similarity = float(np.max(similarities))
        
        if max_similarity >= 0.3:  # 30% semantic similarity threshold
            best_topic = syllabus_topics[np.argmax(similarities)]
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