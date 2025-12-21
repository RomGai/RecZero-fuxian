#!/usr/bin/env python3
"""Debug script for NDCG calculation"""

import sys
sys.path.append('/home/huangyanwen.hyw/code_linlin/RecZero')

from verl.utils.reward_score.sem_seq_rec import get_embedding_calculator, normalize_semantic_id
import numpy as np

def debug_ndcg():
    """Debug the NDCG calculation"""
    
    calculator = get_embedding_calculator()
    calculator._load_data()
    
    # Test semantic IDs
    predicted_semantic = "<a_70> <b_110> <c_69>"
    ground_truth_semantics = ["<a_70> <b_110> <c_69>"]
    
    print(f"Predicted: {predicted_semantic}")
    print(f"Ground truth: {ground_truth_semantics}")
    
    # Get predicted embedding
    pred_embedding = calculator.semantic_id_to_embedding(predicted_semantic)
    print(f"Predicted embedding shape: {pred_embedding.shape}")
    
    # Calculate similarities with all items using inner product
    similarities = np.dot(calculator._embeddings, pred_embedding)
    print(f"Similarities shape: {similarities.shape}")
    print(f"Max similarity: {similarities.max()}")
    print(f"Min similarity: {similarities.min()}")
    
    # Get top-k most similar items
    k = 1000
    top_k_indices = np.argsort(similarities)[::-1][:k]
    print(f"Top 10 indices: {top_k_indices[:10]}")
    print(f"Top 10 similarities: {similarities[top_k_indices[:10]]}")
    
    # Check if predicted item is in top results
    pred_tokens = ["<a_70>", "<b_110>", "<c_69>"]
    pred_tuple = tuple(pred_tokens)
    pred_item_id = calculator._semantic2id.get(pred_tuple)
    print(f"Predicted item ID: {pred_item_id}")
    
    if pred_item_id is not None:
        pred_rank = np.where(top_k_indices == pred_item_id)[0]
        if len(pred_rank) > 0:
            print(f"Predicted item rank in top-{k}: {pred_rank[0] + 1}")
            print(f"Predicted item similarity: {similarities[pred_item_id]}")
        else:
            print(f"Predicted item not in top-{k}")
    
    # Check relevance scores
    relevance_scores = []
    for i, idx in enumerate(top_k_indices[:10]):  # Check first 10
        item_semantic = calculator._id2semantic[str(idx)]
        item_semantic_str = ' '.join(item_semantic)
        
        # Check if this item is in ground truth
        is_relevant = any(
            normalize_semantic_id(item_semantic_str) == normalize_semantic_id(gt) 
            for gt in ground_truth_semantics
        )
        relevance_scores.append(1.0 if is_relevant else 0.0)
        
        if i < 5:  # Print first 5 for debugging
            print(f"Rank {i+1}: ID={idx}, Semantic={item_semantic}, Relevant={is_relevant}")
    
    print(f"First 10 relevance scores: {relevance_scores}")
    
    # Calculate DCG manually
    dcg = 0.0
    for i, relevance in enumerate(relevance_scores):
        if relevance > 0:
            dcg += relevance / np.log2(i + 2)
            print(f"DCG contribution at rank {i+1}: {relevance / np.log2(i + 2)}")
    
    print(f"DCG: {dcg}")

if __name__ == "__main__":
    debug_ndcg()