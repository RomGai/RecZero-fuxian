#!/usr/bin/env python3
"""Check embedding organization and index mapping"""

import sys
sys.path.append('/home/huangyanwen.hyw/code_linlin/RecZero')

import json
import numpy as np

def check_embedding_organization():
    """Check how embeddings are organized"""
    
    # Load embeddings
    emb_path = "/home/huangyanwen.hyw/code_linlin/RecZero/data/Sports/Sports.emb-qwen-td.npy"
    embeddings = np.load(emb_path)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")
    
    # Check for nan/inf values
    nan_count = np.isnan(embeddings).sum()
    inf_count = np.isinf(embeddings).sum()
    print(f"NaN values: {nan_count}")
    print(f"Inf values: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        print("WARNING: Embeddings contain NaN or Inf values!")
    
    # Load index mapping
    index_path = "/home/huangyanwen.hyw/code_linlin/RecZero/data/Sports/Sports.index.json"
    with open(index_path, 'r') as f:
        id2semantic = json.load(f)
    
    print(f"Index mapping size: {len(id2semantic)}")
    print(f"Expected embedding size: {len(id2semantic)}")
    print(f"Actual embedding size: {embeddings.shape[0]}")
    
    # Check if sizes match
    if len(id2semantic) != embeddings.shape[0]:
        print("ERROR: Size mismatch between embeddings and index mapping!")
        return
    
    # Check some specific mappings
    print("\nChecking specific mappings:")
    for i in [0, 1, 2, 100, 1000]:
        if str(i) in id2semantic:
            semantic_id = id2semantic[str(i)]
            embedding = embeddings[i]
            print(f"Item {i}: semantic={semantic_id}, embedding_norm={np.linalg.norm(embedding)}")
            
            # Check for specific embedding issues
            if np.isnan(embedding).any():
                print(f"  WARNING: Item {i} embedding contains NaN")
            if np.isinf(embedding).any():
                print(f"  WARNING: Item {i} embedding contains Inf")
    
    # Test self-similarity
    print("\nTesting self-similarity:")
    test_embedding = embeddings[0]
    self_similarity = np.dot(test_embedding, test_embedding)
    print(f"Self-similarity of item 0: {self_similarity}")
    
    # Test similarity with other items
    similarities = np.dot(embeddings, test_embedding)
    print(f"Similarities shape: {similarities.shape}")
    print(f"Max similarity: {similarities.max()}")
    print(f"Min similarity: {similarities.min()}")
    print(f"Item 0 similarity rank: {np.argsort(similarities)[::-1].tolist().index(0) + 1}")
    
    # Check if embeddings are normalized
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nEmbedding norms - Min: {norms.min():.4f}, Max: {norms.max():.4f}, Mean: {norms.mean():.4f}")
    
    return embeddings, id2semantic

if __name__ == "__main__":
    check_embedding_organization()