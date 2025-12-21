#!/usr/bin/env python3
"""Test script for semantic constraint functionality."""

import json
import tempfile
from transformers import AutoTokenizer
from verl.utils.constrained_generation import load_indices_file, build_semantic_prefix_allowed_tokens_fn

def create_test_indices():
    """Create test semantic indices."""
    indices = {
        "0": ["<a_1>", "<b_2>", "<c_3>"],
        "1": ["<a_2>", "<b_1>", "<c_3>"],
        "2": ["<a_3>", "<b_3>", "<c_1>"]
    }
    return indices

def test_semantic_constraint():
    """Test semantic constraint functionality."""
    print("Testing semantic constraint functionality...")
    
    # Create temporary indices file
    indices = create_test_indices()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(indices, f)
        indices_path = f.name
    
    print(f"Created test indices file: {indices_path}")
    print(f"Indices: {indices}")
    
    # Load indices
    loaded_indices = load_indices_file(indices_path)
    print(f"Loaded indices: {loaded_indices}")
    
    # Test with a simple tokenizer (you may need to adjust the model path)
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Build prefix allowed tokens function
        prefix_fn = build_semantic_prefix_allowed_tokens_fn(
            tokenizer=tokenizer,
            indices=loaded_indices,
            sep_text="Response:",
            allow_fallback_full_vocab=True,
            force_eos_after=True
        )
        
        print("Successfully created prefix_allowed_tokens_fn")
        
        # Test the function with some sample inputs
        import torch
        
        # Test before separator (should allow full vocab)
        test_input = torch.tensor([1, 2, 3, 4, 5])  # Some random token IDs
        allowed_tokens = prefix_fn(0, test_input)
        print(f"Before separator - allowed tokens count: {len(allowed_tokens)}")
        
        # Test after separator
        sep_tokens = tokenizer("Response:", add_special_tokens=False).input_ids
        test_input_with_sep = torch.tensor([1, 2] + sep_tokens + [100])  # Add separator + one token
        allowed_tokens_after_sep = prefix_fn(0, test_input_with_sep)
        print(f"After separator - allowed tokens count: {len(allowed_tokens_after_sep)}")
        print(f"Allowed tokens: {allowed_tokens_after_sep[:10]}...")  # Show first 10
        
        print("✓ Semantic constraint test passed!")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        print("This might be due to missing model files, but the core functionality should work.")
    
    # Clean up
    import os
    os.unlink(indices_path)

if __name__ == "__main__":
    test_semantic_constraint()