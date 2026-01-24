"""
FLAN-T5 Based Syllabus Relevance Checker
Context-aware validation using instruction-following language model
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
from typing import Dict, Tuple
import json

class FLANT5RelevanceChecker:
    def __init__(self):
        # Load FLAN-T5-Small model
        self.model_name = "google/flan-t5-small"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        
        # Load syllabus context
        self.syllabus_context = self._load_syllabus_context()
        
        print("FLAN-T5 relevance checker loaded")
    
    def _load_syllabus_context(self) -> str:
        """Load Computer Networks syllabus context"""
        return """
        Computer Networks Syllabus:
        Unit 1: Introduction - Network types, protocols, OSI model, TCP/IP architecture
        Unit 2: Physical & Data Link Layer - Transmission media, Ethernet, error detection, flow control
        Unit 3: Network Layer - IP addressing, routing algorithms, subnetting, ICMP, NAT, DHCP
        Unit 4: Transport Layer - TCP, UDP, connection management, congestion control, reliability
        Unit 5: Application Layer - HTTP, DNS, FTP, SMTP, network management protocols
        """
    
    def _create_classification_prompt(self, question: str) -> str:
        """Create structured prompt for relevance classification"""
        prompt = f"""Classify this student question about Computer Networks syllabus.

Syllabus covers: OSI model, TCP/IP, routing, Ethernet, HTTP, DNS, network protocols, three-way handshake, congestion control.

Question: "{question}"

Is this question:
A) VALID - directly covered in Computer Networks syllabus (TCP, UDP, HTTP, routing, OSI, handshake, etc.)
B) WARNING - related to networking but advanced/peripheral
C) REJECTED - not about computer networking

Answer with just the letter (A, B, or C):"""
        return prompt
    
    def check_relevance_with_flan_t5(self, question: str) -> Tuple[float, Dict]:
        """Use FLAN-T5 for contextual relevance checking with secondary validation"""
        
        # Primary check: Explicit non-networking terms
        non_networking_terms = ['python', 'java', 'programming', 'fear', 'emotion', 'psychology', 'cooking', 'sports', 'linkedin', 'facebook', 'twitter', 'social', 'career', 'job', 'professional']
        social_networking_patterns = ['networking in', 'networking on', 'professional networking', 'business networking', 'career networking']
        
        question_lower = question.lower()
        
        # Check for social/professional networking context
        for pattern in social_networking_patterns:
            if pattern in question_lower:
                return 0.0, {
                    'classification': 'REJECTED',
                    'confidence': 'HIGH',
                    'reasoning': f'Detected social/professional networking context: "{pattern}"',
                    'model_response': 'Pre-filtered',
                    'method': 'Pattern matching (pre-filter)'
                }
        
        # Check for other non-networking terms
        for term in non_networking_terms:
            if term in question_lower:
                return 0.0, {
                    'classification': 'REJECTED',
                    'confidence': 'HIGH',
                    'reasoning': f'Contains non-networking term: "{term}"',
                    'model_response': 'Pre-filtered',
                    'method': 'Term filtering (pre-filter)'
                }
        
        # Secondary check: FLAN-T5 classification
        # Create classification prompt
        prompt = self._create_classification_prompt(question)
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=150,
                temperature=0.3,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Parse model response
        classification, confidence, reasoning = self._parse_model_response(response)
        
        # Convert to numerical score
        score = self._classification_to_score(classification, confidence)
        
        return score, {
            'classification': classification,
            'confidence': confidence,
            'reasoning': reasoning,
            'model_response': response,
            'method': 'FLAN-T5 contextual analysis'
        }
    
    def _parse_model_response(self, response: str) -> Tuple[str, str, str]:
        """Parse FLAN-T5 response into components"""
        
        response = response.strip().upper()
        
        # Map letter responses to classifications
        if 'A' in response:
            classification = "VALID"
            confidence = "HIGH"
            reasoning = "Question directly covered in Computer Networks syllabus"
        elif 'B' in response:
            classification = "WARNING"
            confidence = "MEDIUM"
            reasoning = "Question related to networking but may be advanced or peripheral"
        elif 'C' in response:
            classification = "REJECTED"
            confidence = "HIGH"
            reasoning = "Question not related to computer networking"
        else:
            # Fallback: Check for networking keywords
            networking_keywords = ['tcp', 'udp', 'http', 'ip', 'osi', 'routing', 'ethernet', 'dns', 'handshake', 'protocol', 'network']
            response_lower = response.lower()
            
            # If response contains networking terms, assume VALID
            if any(keyword in response_lower for keyword in networking_keywords):
                classification = "VALID"
                confidence = "MEDIUM"
                reasoning = "Fallback: Contains networking terminology"
            else:
                classification = "WARNING"
                confidence = "LOW"
                reasoning = f"Unclear model response: {response}"
        
        return classification, confidence, reasoning
    
    def _classification_to_score(self, classification: str, confidence: str) -> float:
        """Convert classification and confidence to numerical score"""
        
        base_scores = {
            'VALID': 0.9,
            'WARNING': 0.5,  # Reduced from 0.6 to be more conservative
            'REJECTED': 0.1
        }
        
        confidence_multipliers = {
            'HIGH': 1.0,
            'MEDIUM': 0.8,  # Reduced from 0.85
            'LOW': 0.6      # Reduced from 0.7
        }
        
        base_score = base_scores.get(classification, 0.1)
        multiplier = confidence_multipliers.get(confidence, 0.6)
        
        return base_score * multiplier

def test_flan_t5_checker():
    """Test FLAN-T5 based relevance checking"""
    checker = FLANT5RelevanceChecker()
    
    test_questions = [
        "What is TCP three-way handshake?",
        "How does HTTP work?", 
        "What is machine learning?",
        "networking in linkedin",
        "Explain BGP route optimization"
    ]
    
    print("\n=== FLAN-T5 Relevance Checking Results ===\n")
    
    for question in test_questions:
        score, details = checker.check_relevance_with_flan_t5(question)
        
        print(f"Question: '{question}'")
        print(f"Classification: {details['classification']}")
        print(f"Score: {score:.3f}")
        print(f"Confidence: {details['confidence']}")
        print(f"Reasoning: {details['reasoning']}")
        print("-" * 60)

if __name__ == "__main__":
    test_flan_t5_checker()