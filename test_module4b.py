"""
Test Module 4B: Fine-tuning Validation
Validates the fine-tuning process and saved model.
"""

import os
import json
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

def test_model_files():
    """Test if all required model files are saved"""
    print("Testing saved model files...")
    
    model_dir = "models/flan_t5_mcp"
    required_files = [
        "config.json",
        "model.safetensors", 
        "tokenizer_config.json",
        "tokenizer.json"
    ]
    
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        assert os.path.exists(file_path), f"Missing file: {file}"
    
    print("[PASS] All required model files exist")
    return True

def test_model_loading():
    """Test loading the fine-tuned model"""
    print("Testing model loading...")
    
    model_dir = "models/flan_t5_mcp"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    
    assert tokenizer is not None, "Failed to load tokenizer"
    assert model is not None, "Failed to load model"
    
    print("[PASS] Fine-tuned model loads successfully")
    return model, tokenizer

def test_dataset_loading():
    """Test MCP dataset loading"""
    print("Testing dataset loading...")
    
    train_file = "data/processed/mcp_train.jsonl"
    val_file = "data/processed/mcp_val.jsonl"
    
    # Count training samples
    train_count = 0
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            assert 'input' in sample, "Missing 'input' field"
            assert 'output' in sample, "Missing 'output' field"
            train_count += 1
    
    # Count validation samples
    val_count = 0
    with open(val_file, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            assert 'input' in sample, "Missing 'input' field"
            assert 'output' in sample, "Missing 'output' field"
            val_count += 1
    
    print(f"[PASS] Dataset loaded: {train_count} train, {val_count} val samples")
    return train_count, val_count

def test_model_inference():
    """Test basic inference with fine-tuned model"""
    print("Testing model inference...")
    
    model_dir = "models/flan_t5_mcp"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    
    # Test input in MCP format
    test_input = """[QUESTION]
What is a network protocol?

[CONTEXT]
A protocol defines the format and the order of messages exchanged between two or more communicating entities.

[INSTRUCTION]
Answer the question academically using only the given context."""
    
    # Generate response
    inputs = tokenizer(test_input, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=100,
            num_beams=2,
            early_stopping=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    assert len(response) > 0, "Model generated empty response"
    print(f"[PASS] Model inference working")
    print(f"Sample response: {response[:100]}...")
    
    return response

def run_all_tests():
    """Run all Module 4B validation tests"""
    print("=" * 50)
    print("MODULE 4B VALIDATION TESTS")
    print("=" * 50)
    
    try:
        # Test 1: Model files
        test_model_files()
        
        # Test 2: Model loading
        model, tokenizer = test_model_loading()
        
        # Test 3: Dataset loading
        train_count, val_count = test_dataset_loading()
        
        # Test 4: Model inference
        response = test_model_inference()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED [SUCCESS]")
        print("Module 4B fine-tuning completed successfully!")
        print(f"Training samples: {train_count}")
        print(f"Validation samples: {val_count}")
        print(f"Model saved to: models/flan_t5_mcp/")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        print("=" * 50)
        raise

if __name__ == "__main__":
    run_all_tests()