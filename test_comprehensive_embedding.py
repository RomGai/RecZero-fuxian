#!/usr/bin/env python3
"""Comprehensive test script for the embedding-based reward system"""

import sys
import random
import json
sys.path.append('/home/huangyanwen.hyw/code_linlin/RecZero')

from verl.utils.reward_score.sem_seq_rec import compute_sem_seq_rec_embedding_score, get_embedding_calculator

def load_test_data():
    """Load some real semantic IDs for testing"""
    index_path = "/home/huangyanwen.hyw/code_linlin/RecZero/data/Sports/Sports.index.json"
    with open(index_path, 'r') as f:
        id2semantic = json.load(f)
    
    # Get a sample of semantic IDs
    semantic_ids = []
    for i in range(min(50, len(id2semantic))):
        semantic_id = ' '.join(id2semantic[str(i)])
        semantic_ids.append(semantic_id)
    
    return semantic_ids

def test_comprehensive_embedding_reward():
    """Test the embedding reward system with multiple cases"""
    
    print("Loading test data...")
    semantic_ids = load_test_data()
    print(f"Loaded {len(semantic_ids)} test semantic IDs")
    
    calculator = get_embedding_calculator()
    
    test_cases = []
    
    # Test Case 1: Exact matches (should return 1.0)
    print("\n=== Test Case 1: Exact Matches ===")
    exact_match_scores = []
    for i, semantic_id in enumerate(semantic_ids[:10]):
        solution_str = f"""
        Human: What's the next item recommendation?
        Assistant: Based on the sequence, I recommend:
        <answer>{semantic_id}</answer>
        """
        
        score = compute_sem_seq_rec_embedding_score(solution_str, semantic_id)
        exact_match_scores.append(score)
        
        if i < 3:  # Print first 3 for debugging
            print(f"Test {i+1}: Semantic ID = {semantic_id}")
            print(f"         Score = {score:.4f}")
        
        test_cases.append({
            'type': 'exact_match',
            'predicted': semantic_id,
            'ground_truth': semantic_id,
            'score': score
        })
    
    print(f"Exact match scores: min={min(exact_match_scores):.4f}, max={max(exact_match_scores):.4f}, avg={sum(exact_match_scores)/len(exact_match_scores):.4f}")
    
    # Test Case 2: Different predictions (should return < 1.0)
    print("\n=== Test Case 2: Different Predictions ===")
    different_scores = []
    for i in range(10):
        predicted_semantic = semantic_ids[i]
        ground_truth_semantic = semantic_ids[(i + 5) % len(semantic_ids)]  # Different semantic ID
        
        solution_str = f"""
        Assistant: <answer>{predicted_semantic}</answer>
        """
        
        score = compute_sem_seq_rec_embedding_score(solution_str, ground_truth_semantic)
        different_scores.append(score)
        
        if i < 3:  # Print first 3 for debugging
            print(f"Test {i+1}: Predicted = {predicted_semantic}")
            print(f"         Ground truth = {ground_truth_semantic}")
            print(f"         Score = {score:.4f}")
        
        test_cases.append({
            'type': 'different_prediction',
            'predicted': predicted_semantic,
            'ground_truth': ground_truth_semantic,
            'score': score
        })
    
    print(f"Different prediction scores: min={min(different_scores):.4f}, max={max(different_scores):.4f}, avg={sum(different_scores)/len(different_scores):.4f}")
    
    # Test Case 3: Invalid predictions (should return 0.0)
    print("\n=== Test Case 3: Invalid Predictions ===")
    invalid_predictions = [
        "<a_999999> <b_999999> <c_999999>",  # Non-existent semantic ID
        "<a_1> <b_2>",  # Incomplete semantic ID
        "random text",  # No semantic ID pattern
        "",  # Empty
    ]
    
    invalid_scores = []
    for i, invalid_pred in enumerate(invalid_predictions):
        ground_truth = semantic_ids[0]
        
        solution_str = f"""
        Assistant: <answer>{invalid_pred}</answer>
        """
        
        score = compute_sem_seq_rec_embedding_score(solution_str, ground_truth)
        invalid_scores.append(score)
        
        print(f"Test {i+1}: Invalid prediction = '{invalid_pred}'")
        print(f"         Score = {score:.4f}")
        
        test_cases.append({
            'type': 'invalid_prediction',
            'predicted': invalid_pred,
            'ground_truth': ground_truth,
            'score': score
        })
    
    # Test Case 4: Multiple ground truths
    print("\n=== Test Case 4: Multiple Ground Truths ===")
    multi_gt_scores = []
    for i in range(5):
        predicted_semantic = semantic_ids[i]
        ground_truths = [semantic_ids[i], semantic_ids[i+10], semantic_ids[i+20]]  # Include the predicted one
        
        solution_str = f"""
        Assistant: <answer>{predicted_semantic}</answer>
        """
        
        score = compute_sem_seq_rec_embedding_score(solution_str, ground_truths)
        multi_gt_scores.append(score)
        
        print(f"Test {i+1}: Predicted = {predicted_semantic}")
        print(f"         Ground truths = {ground_truths}")
        print(f"         Score = {score:.4f}")
        
        test_cases.append({
            'type': 'multiple_ground_truth',
            'predicted': predicted_semantic,
            'ground_truth': ground_truths,
            'score': score
        })
    
    # Test Case 5: No answer extraction
    print("\n=== Test Case 5: No Answer Extraction ===")
    no_answer_solutions = [
        "Human: What's the recommendation? Assistant: I don't know.",
        "Assistant: The item is good but I can't specify.",
        "Some random text without answer tags",
    ]
    
    no_answer_scores = []
    for i, solution in enumerate(no_answer_solutions):
        ground_truth = semantic_ids[0]
        
        score = compute_sem_seq_rec_embedding_score(solution, ground_truth)
        no_answer_scores.append(score)
        
        print(f"Test {i+1}: No answer solution")
        print(f"         Score = {score:.4f}")
        
        test_cases.append({
            'type': 'no_answer',
            'predicted': 'N/A',
            'ground_truth': ground_truth,
            'score': score
        })
    
    # Summary
    print("\n=== Summary ===")
    print(f"Total test cases: {len(test_cases)}")
    
    # Check if exact matches return 1.0
    exact_match_perfect = sum(1 for case in test_cases if case['type'] == 'exact_match' and case['score'] == 1.0)
    exact_match_total = sum(1 for case in test_cases if case['type'] == 'exact_match')
    print(f"Exact matches returning 1.0: {exact_match_perfect}/{exact_match_total}")
    
    # Check if invalid predictions return 0.0
    invalid_zero = sum(1 for case in test_cases if case['type'] == 'invalid_prediction' and case['score'] == 0.0)
    invalid_total = sum(1 for case in test_cases if case['type'] == 'invalid_prediction')
    print(f"Invalid predictions returning 0.0: {invalid_zero}/{invalid_total}")
    
    # Check if no answer cases return 0.0
    no_answer_zero = sum(1 for case in test_cases if case['type'] == 'no_answer' and case['score'] == 0.0)
    no_answer_total = sum(1 for case in test_cases if case['type'] == 'no_answer')
    print(f"No answer cases returning 0.0: {no_answer_zero}/{no_answer_total}")
    
    # Overall statistics
    all_scores = [case['score'] for case in test_cases]
    print(f"Overall score range: [{min(all_scores):.4f}, {max(all_scores):.4f}]")
    print(f"Average score: {sum(all_scores)/len(all_scores):.4f}")
    
    return test_cases

if __name__ == "__main__":
    try:
        test_cases = test_comprehensive_embedding_reward()
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()