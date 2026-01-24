#!/usr/bin/env python3
"""
Combined Test Module for Module 4 (4A, 4B, 4C, 4D)
Runs baseline inference, fine-tuned model checks, grounded inference, and comparison flows in sequence.
"""

import os
import json
import sys
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.module2_semantic_indexing.semantic_indexing import SemanticIndexer


def load_sample_question():
    test_file = "data/processed/mcp_test.jsonl"
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"MCP test file not found at {test_file}")

    with open(test_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        sample = json.loads(first_line)
    return sample


def test_sample_loading():
    print("Testing sample question loading...")
    sample = load_sample_question()

    assert 'input' in sample, "Sample should contain 'input' field"
    assert '[QUESTION]' in sample['input'], "Input should contain [QUESTION] section"
    assert '[CONTEXT]' in sample['input'], "Input should contain [CONTEXT] section"
    assert '[INSTRUCTION]' in sample['input'], "Input should contain [INSTRUCTION] section"

    print("[PASS] Sample loading test passed")
    return sample


def test_context_retrieval():
    print("Testing context retrieval (FAISS)...")
    indexer = SemanticIndexer()
    indexer.load_index()

    sample = load_sample_question()
    question = sample['input'].split('[QUESTION]\n')[1].split('\n\n[CONTEXT]')[0]

    results = indexer.search(question, k=4)
    assert isinstance(results, list), "Search should return a list"
    assert len(results) > 0, "Search should return at least one result"

    print(f"[PASS] Context retrieval returned {len(results)} chunks")
    return results


def test_model_files():
    print("Testing fine-tuned model files exist...")
    model_dir = "models/flan_t5_mcp"
    required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
        "tokenizer.json"
    ]

    missing = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
    assert not missing, f"Missing model files: {missing}"

    print("[PASS] All required fine-tuned model files exist")


def test_model_loading():
    print("Testing baseline and fine-tuned model loading...")

    # Baseline
    baseline_name = "google/flan-t5-small"
    baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_name)
    baseline_model = T5ForConditionalGeneration.from_pretrained(baseline_name)
    assert baseline_tokenizer is not None and baseline_model is not None, "Baseline model failed to load"

    # Fine-tuned
    ft_dir = "models/flan_t5_mcp"
    ft_tokenizer = AutoTokenizer.from_pretrained(ft_dir)
    ft_model = T5ForConditionalGeneration.from_pretrained(ft_dir)
    assert ft_tokenizer is not None and ft_model is not None, "Fine-tuned model failed to load"

    print("[PASS] Both baseline and fine-tuned models loaded successfully")
    return (baseline_model, baseline_tokenizer, ft_model, ft_tokenizer)


def generate_answer(model, tokenizer, mcp_input, max_length=200):
    inputs = tokenizer(mcp_input, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def test_answer_generation(baseline_model, baseline_tokenizer, ft_model, ft_tokenizer):
    print("Testing answer generation for both baseline and fine-tuned models...")

    test_input = """[QUESTION]
What is a network protocol?

[CONTEXT]
A protocol defines the format and the order of messages exchanged between two or more communicating entities.

[INSTRUCTION]
Answer the question academically using only the given context."""

    bl_answer = generate_answer(baseline_model, baseline_tokenizer, test_input, max_length=50)
    ft_answer = generate_answer(ft_model, ft_tokenizer, test_input, max_length=50)

    assert isinstance(bl_answer, str) and len(bl_answer) > 0, "Baseline answer should be non-empty string"
    assert isinstance(ft_answer, str) and len(ft_answer) > 0, "Fine-tuned answer should be non-empty string"

    print("[PASS] Answer generation working")
    print(f"Baseline answer sample: {bl_answer}")
    print(f"Fine-tuned answer sample: {ft_answer}")

    return bl_answer, ft_answer


def test_model_difference(bl_answer, ft_answer):
    print("Testing whether fine-tuning changed behavior (string comparison)...")

    if bl_answer.strip().lower() == ft_answer.strip().lower():
        print("[INFO] Answers are identical — fine-tuning may not have changed this simple case")
    else:
        print("[PASS] Answers differ — fine-tuning effect observed")


def run_all_tests():
    print("=" * 60)
    print("MODULE 4 COMBINED TESTS")
    print("=" * 60)

    try:
        sample = test_sample_loading()
        test_context_retrieval()
        test_model_files()
        baseline_model, baseline_tokenizer, ft_model, ft_tokenizer = test_model_loading()
        bl_answer, ft_answer = test_answer_generation(baseline_model, baseline_tokenizer, ft_model, ft_tokenizer)
        test_model_difference(bl_answer, ft_answer)

        print("\nALL MODULE 4 TESTS PASSED [SUCCESS]")

    except AssertionError as e:
        print(f"[FAIL] Test failed: {e}")
        raise
    except Exception as e:
        print(f"[ERROR] Unexpected error during testing: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
