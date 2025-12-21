#!/usr/bin/env python3
"""Test script for the embedding-based reward system"""

import sys
sys.path.append('/home/huangyanwen.hyw/code_linlin/RecZero')

from verl.utils.reward_score.sem_seq_rec import compute_sem_seq_rec_embedding_score, get_embedding_calculator

def test_embedding_reward():
    """Test the embedding reward system"""
    
    # Sample solution from LLM (with semantic ID)
    solution_str = """
    Human: What's the next item recommendation?
    
    Assistant: Based on the sequence, I recommend:
    <answer><a_70> <b_110> <c_69></answer>
    """
    
    # Ground truth semantic ID  
    ground_truth = "<a_70> <b_110> <c_69>"  # Same as prediction for testing
    
    print("Testing embedding-based reward system...")
    print(f"Solution: {solution_str.strip()}")
    print(f"Ground truth: {ground_truth}")
    
    # Test the reward calculation
    try:
        score = compute_sem_seq_rec_embedding_score(solution_str, ground_truth)
        print(f"NDCG@1000 Score: {score:.4f}")
        
        # Test with different prediction
        different_solution = """
        Assistant: <answer><a_70> <b_110> <c_29></answer>
        """
        
        score2 = compute_sem_seq_rec_embedding_score(different_solution, ground_truth)
        print(f"Different prediction NDCG@1000 Score: {score2:.4f}")
        
        # Test embedding lookup directly
        calculator = get_embedding_calculator()
        embedding = calculator.semantic_id_to_embedding("<a_70> <b_110> <c_69>")
        if embedding is not None:
            print(f"Successfully retrieved embedding shape: {embedding.shape}")
        else:
            print("Failed to retrieve embedding")
            
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_embedding_reward()