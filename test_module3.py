"""
Test script for Module 3: MCP Dataset Generation
Validates the generated MCP dataset format and content.
"""

import json
import os
from modules.module3_qa_system.mcp_dataset_generator import MCPDatasetGenerator

def test_mcp_generation():
    """Test MCP dataset generation with a small sample."""
    print("=== Testing MCP Dataset Generation ===")
    
    # Initialize generator
    generator = MCPDatasetGenerator()
    
    # Test retrieval for a sample question
    sample_question = "Compare the functionality of a hub and a switch in relation to collision domains"
    print(f"\nTesting retrieval for: {sample_question}")
    
    # Retrieve context
    context_chunks = generator.retrieve_context(sample_question, k=3)
    print(f"Retrieved {len(context_chunks)} chunks:")
    
    for i, chunk in enumerate(context_chunks):
        print(f"  {i+1}. Score: {chunk['score']:.4f}")
        print(f"     Source: {chunk['source_file']}")
        print(f"     Text: {chunk['text'][:100]}...")
        print()
    
    # Test MCP input creation
    mcp_input = generator.create_mcp_input(sample_question, context_chunks)
    print("Generated MCP Input:")
    print("=" * 50)
    print(mcp_input)
    print("=" * 50)
    
    return True

def validate_dataset_files():
    """Validate the generated JSONL files."""
    print("\n=== Validating Generated Dataset Files ===")
    
    files_to_check = [
        "data/processed/mcp_train.jsonl",
        "data/processed/mcp_val.jsonl", 
        "data/processed/mcp_test.jsonl",
        "data/processed/mcp_dataset_summary.json"
    ]
    
    for filepath in files_to_check:
        if os.path.exists(filepath):
            print(f"+ {filepath} exists")
            
            if filepath.endswith('.jsonl'):
                # Count lines and validate format
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    print(f"  - Contains {len(lines)} examples")
                    
                    # Validate first example
                    if lines:
                        try:
                            example = json.loads(lines[0])
                            required_fields = ['input', 'output', 'metadata']
                            for field in required_fields:
                                if field in example:
                                    print(f"  - + Has '{field}' field")
                                else:
                                    print(f"  - - Missing '{field}' field")
                        except json.JSONDecodeError:
                            print(f"  - - Invalid JSON format")
            
            elif filepath.endswith('.json'):
                # Validate summary file
                with open(filepath, 'r') as f:
                    summary = json.load(f)
                    print(f"  - Total examples: {summary.get('total_examples', 'N/A')}")
                    print(f"  - Split ratio: {summary.get('split_ratio', 'N/A')}")
        else:
            print(f"- {filepath} not found")
    
    return True

def show_sample_examples():
    """Display sample examples from each split."""
    print("\n=== Sample Examples from Each Split ===")
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        filepath = f"data/processed/mcp_{split}.jsonl"
        if os.path.exists(filepath):
            print(f"\n--- {split.upper()} Split Sample ---")
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if first_line:
                    example = json.loads(first_line)
                    print(f"Input preview: {example['input'][:200]}...")
                    print(f"Output preview: {example['output'][:200]}...")
                    print(f"Topic: {example['metadata']['topic']}")
                    print(f"Retrieved chunks: {len(example['metadata']['retrieved_chunks'])}")

def main():
    """Run all tests."""
    try:
        # Test MCP generation
        test_mcp_generation()
        
        # Validate files (if they exist)
        validate_dataset_files()
        
        # Show samples (if files exist)
        show_sample_examples()
        
        print("\n=== All Tests Completed ===")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()