#!/usr/bin/env python3
"""
Test script for Module 1: Knowledge Ingestion & Preparation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.module1_knowledge_ingestion.knowledge_ingestion import KnowledgeIngestionPipeline, DocumentChunk

def test_text_cleaning():
    """Test text cleaning functionality"""
    pipeline = KnowledgeIngestionPipeline("data/raw/syllabus/sample_syllabus.txt")
    
    # Test text with noise
    noisy_text = """
    HEADER TEXT
    
    This is a paragraph with content.
    
    123
    
    Another paragraph here.
    
    FOOTER TEXT
    """
    
    cleaned = pipeline._clean_text(noisy_text)
    print("Text Cleaning Test:")
    print(f"Original length: {len(noisy_text)}")
    print(f"Cleaned length: {len(cleaned)}")
    print(f"Cleaned text: {cleaned[:100]}...")
    print()

def test_chunking():
    """Test paragraph chunking"""
    pipeline = KnowledgeIngestionPipeline("data/raw/syllabus/sample_syllabus.txt")
    
    sample_text = """
    Computer networks are interconnected systems that allow devices to communicate and share resources. Networks can be classified based on their geographical coverage, such as Local Area Networks (LANs), Wide Area Networks (WANs), and Metropolitan Area Networks (MANs).

    The OSI model is a conceptual framework that standardizes the functions of a telecommunication or computing system. It consists of seven layers: Physical, Data Link, Network, Transport, Session, Presentation, and Application layers.

    TCP (Transmission Control Protocol) is a connection-oriented protocol that provides reliable data transmission. It ensures that data packets are delivered in the correct order and without errors.
    """
    
    chunks = pipeline._chunk_by_paragraphs(sample_text)
    print("Chunking Test:")
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} ({len(chunk)} chars): {chunk[:50]}...")
    print()

def test_unit_assignment():
    """Test unit assignment based on keywords"""
    pipeline = KnowledgeIngestionPipeline("data/raw/syllabus/sample_syllabus.txt")
    
    test_chunks = [
        "Computer networks are fundamental systems for communication...",
        "TCP and UDP are transport layer protocols that handle data transmission...",
        "Network security involves encryption and authentication mechanisms..."
    ]
    
    print("Unit Assignment Test:")
    for chunk in test_chunks:
        unit = pipeline._assign_unit(chunk)
        print(f"Text: {chunk[:50]}...")
        print(f"Assigned Unit: {unit}")
        print()

def create_sample_chunk():
    """Create a sample DocumentChunk for testing"""
    chunk = DocumentChunk(
        text="This is a sample chunk about network protocols.",
        unit="Unit 2: Data Link and Network Layer",
        source_type="textbook",
        book_priority=1,
        source_file="sample_textbook",
        chunk_id="sample_textbook_chunk_1"
    )
    
    print("Sample DocumentChunk:")
    print(f"Text: {chunk.text}")
    print(f"Unit: {chunk.unit}")
    print(f"Source Type: {chunk.source_type}")
    print(f"Priority: {chunk.book_priority}")
    print(f"Chunk ID: {chunk.chunk_id}")
    print()

if __name__ == "__main__":
    print("=== MODULE 1 TESTING ===\n")
    
    test_text_cleaning()
    test_chunking()
    test_unit_assignment()
    create_sample_chunk()
    
    print("=== TESTING COMPLETE ===")
    print("Module 1 is ready for integration with Module 2!")