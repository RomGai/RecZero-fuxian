import re
import json
import os
import numpy as np
from typing import Dict, Tuple, Optional, Union, List, Callable

def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extract semantic ID from solution string
    
    Args:
        solution_str: Raw solution string from model output
        
    Returns:
        Tuple of (extracted semantic ID, processed string)
    """
    # Extract assistant response
    # print("solution_str: ", solution_str)
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[Warning] Could not find assistant response markers")
        return None, solution_str

    # Try different patterns to extract the answer
    patterns = [
        r'<answer>(.*?)</answer>',  # XML tags
        r'<semantic_id>(.*?)</semantic_id>',  # Semantic ID tags
        r'<prediction>(.*?)</prediction>',  # Prediction tags
        r'(\<a_\d+\>\s*\<b_\d+\>\s*\<c_\d+\>)',  # Direct semantic ID pattern
    ]
    
    for pattern in patterns:
        matches = list(re.finditer(pattern, processed_str, re.DOTALL))
        if matches:
            final_answer = matches[-1].group(1).strip()
            return final_answer, processed_str
    
    # Fallback: search for semantic ID pattern directly
    semantic_pattern = r'(\<a_\d+\>\s*\<b_\d+\>\s*\<c_\d+\>)'
    matches = list(re.finditer(semantic_pattern, processed_str))
    if matches:
        final_answer = matches[-1].group(1).strip()
        return final_answer, processed_str
    
    print("[Warning] Could not extract semantic ID from response")
    return None, processed_str

def normalize_semantic_id(semantic_id: str) -> str:
    """Normalize semantic ID string
    
    Args:
        semantic_id: Raw semantic ID string
        
    Returns:
        Normalized semantic ID string
    """
    if not semantic_id:
        return ""
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', semantic_id.strip())
    
    # Normalize spacing between semantic ID tokens
    normalized = re.sub(r'(\<[abc]_\d+\>)\s*(\<[abc]_\d+\>)', r'\1 \2', normalized)
    
    return normalized

def calculate_semantic_similarity(pred_semantic: str, true_semantic: str) -> float:
    """Calculate similarity between predicted and true semantic IDs
    
    Args:
        pred_semantic: Predicted semantic ID
        true_semantic: True semantic ID
        
    Returns:
        Similarity score (0.0 - 1.0)
    """
    pred_normalized = normalize_semantic_id(pred_semantic)
    true_normalized = normalize_semantic_id(true_semantic)
    
    if not pred_normalized or not true_normalized:
        return 0.0
    
    # Exact match
    if pred_normalized == true_normalized:
        return 1.0
    
    # Extract tokens
    pred_tokens = re.findall(r'\<[abc]_\d+\>', pred_normalized)
    true_tokens = re.findall(r'\<[abc]_\d+\>', true_normalized)
    
    if len(pred_tokens) == 0 or len(true_tokens) == 0:
        return 0.0
    
    # Different number of tokens gets low score
    if len(pred_tokens) != len(true_tokens):
        return 0.1
    
    # Calculate position-wise accuracy
    correct_positions = sum(1 for p, t in zip(pred_tokens, true_tokens) if p == t)
    
    # Calculate position-based score
    position_score = correct_positions / len(true_tokens)
    
    return position_score

def compute_score(solution_str: str, ground_truth: Union[str, Dict]) -> Dict:
    """Compute semantic sequence recommendation score
    
    Args:
        solution_str: Raw model solution
        ground_truth: Ground truth answer or semantic ID string/dict
        
    Returns:
        Dictionary containing scoring results
    """
    # Extract predicted answer
    print("solution_str: ", solution_str)
    predicted_answer, processed_response = extract_solution(solution_str)
    print("predicted_answer: ", predicted_answer)
    print("processed_response: ", processed_response)
    # Parse ground_truth
    if isinstance(ground_truth, dict):
        if 'target_semantic_id' in ground_truth:
            true_semantic_id = ground_truth['target_semantic_id']
        elif 'ground_truth' in ground_truth:
            true_semantic_id = ground_truth['ground_truth']
        else:
            true_semantic_id = str(ground_truth)
    else:
        true_semantic_id = str(ground_truth)
    
    # If no prediction extracted, return zero score
    if predicted_answer is None:
        return {
            'score': 0.0,
            'predicted': None,
            'ground_truth': true_semantic_id,
            'exact_match': False,
            'similarity': 0.0,
            'error': 'Failed to extract prediction'
        }
    
    # Calculate similarity
    # similarity_score = calculate_semantic_similarity(predicted_answer, true_semantic_id)
    similarity_score = 0 
    # Check exact match
    exact_match = ((predicted_answer) == (true_semantic_id))
    
    # Final score: full credit for exact match, penalized similarity otherwise
    final_score = 1.0 if exact_match else max(0.0, similarity_score)  # Small penalty for non-exact matches
    print("score: ", final_score)
    print("predicted: ", predicted_answer)
    print("ground_truth: ", true_semantic_id)
    print("similarity: ", similarity_score)
    print("raw_prediction: ", predicted_answer)
    print("\n")
    return {
        'score': final_score,
        'predicted': predicted_answer,
        'ground_truth': true_semantic_id,
        'exact_match': exact_match,
        'similarity': similarity_score,
        'raw_prediction': predicted_answer
    }

class EmbeddingRewardCalculator:
    """Embedding-based reward calculator for semantic sequence recommendation"""
    
    def __init__(self, 
                 embedding_path: str = "/home/huangyanwen.hyw/code_linlin/RecZero/data/Sports/Sports.emb-qwen-td.npy",
                 index_path: str = "/home/huangyanwen.hyw/code_linlin/RecZero/data/Sports/Sports.index.json"):
        """Initialize the embedding reward calculator
        
        Args:
            embedding_path: Path to the item embedding file (.npy)
            index_path: Path to the id2semantic_id mapping file (.json)
        """
        self.embedding_path = embedding_path
        self.index_path = index_path
        self._embeddings = None
        self._id2semantic = None
        self._semantic2id = None
        
    def _load_data(self):
        """Lazy load embeddings and mappings"""
        if self._embeddings is None:
            print(f"Loading embeddings from {self.embedding_path}")
            embeddings = np.load(self.embedding_path)
            
            # Clean embeddings: replace nan/inf values
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
            
            self._embeddings = embeddings
            print(f"Loaded embeddings shape: {self._embeddings.shape}")
            
            # Check for problematic values after cleaning
            nan_count = np.isnan(self._embeddings).sum()
            inf_count = np.isinf(self._embeddings).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"WARNING: Still have {nan_count} NaN and {inf_count} Inf values after cleaning")
            
        if self._id2semantic is None:
            print(f"Loading index mapping from {self.index_path}")
            with open(self.index_path, 'r') as f:
                self._id2semantic = json.load(f)
            
            # Create reverse mapping: semantic_id -> item_id
            self._semantic2id = {}
            for item_id, semantic_ids in self._id2semantic.items():
                # Convert to tuple for hashing (semantic_ids is a list of 3 elements)
                semantic_tuple = tuple(semantic_ids)
                self._semantic2id[semantic_tuple] = int(item_id)
            
            print(f"Loaded {len(self._id2semantic)} item mappings")
    
    def semantic_id_to_embedding(self, semantic_id: str) -> Optional[np.ndarray]:
        """Convert semantic ID string to embedding
        
        Args:
            semantic_id: Semantic ID string like "<a_70> <b_110> <c_69>"
            
        Returns:
            Embedding vector or None if not found
        """
        self._load_data()
        
        # Parse semantic ID tokens
        tokens = re.findall(r'\<[abc]_\d+\>', semantic_id)
        if len(tokens) != 3:
            return None
            
        # Convert to tuple for lookup
        semantic_tuple = tuple(tokens)
        
        # Get item ID
        item_id = self._semantic2id.get(semantic_tuple)
        if item_id is None:
            return None
            
        # Return embedding
        return self._embeddings[item_id]
    
    def calculate_ndcg_at_k(self, predicted_semantic: str, ground_truth_semantics: List[str], k: int = 1000) -> float:
        """Calculate NDCG@k using embedding similarity
        
        Args:
            predicted_semantic: Predicted semantic ID from LLM
            ground_truth_semantics: List of ground truth semantic IDs
            k: Number of top items to consider for NDCG calculation
            
        Returns:
            NDCG@k score
        """
        self._load_data()
        
        # Normalize inputs
        pred_normalized = normalize_semantic_id(predicted_semantic)
        gt_normalized = [normalize_semantic_id(gt) for gt in ground_truth_semantics]
        
        # Check for exact match first
        if pred_normalized in gt_normalized:
            return 1.0
        
        # Get predicted embedding
        pred_embedding = self.semantic_id_to_embedding(predicted_semantic)
        if pred_embedding is None:
            return 0.0
        
        # Calculate similarities with all items using inner product
        similarities = np.dot(self._embeddings, pred_embedding)
        
        # Get ground truth item IDs for faster lookup
        ground_truth_ids = set()
        for gt_semantic in ground_truth_semantics:
            gt_tokens = re.findall(r'\<[abc]_\d+\>', normalize_semantic_id(gt_semantic))
            if len(gt_tokens) == 3:
                gt_tuple = tuple(gt_tokens)
                gt_id = self._semantic2id.get(gt_tuple)
                if gt_id is not None:
                    ground_truth_ids.add(gt_id)
        
        # Get top-k most similar items
        top_k_indices = np.argsort(similarities)[::-1][:k]  # Sort descending, take top k
        
        # Create relevance scores (1 if item is in ground truth, 0 otherwise)
        relevance_scores = []
        for idx in top_k_indices:
            is_relevant = idx in ground_truth_ids
            relevance_scores.append(1.0 if is_relevant else 0.0)
        
        # Calculate DCG
        dcg = 0.0
        for i, relevance in enumerate(relevance_scores):
            if relevance > 0:
                dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG) - assume best possible ranking
        num_relevant = len(ground_truth_ids)
        idcg = 0.0
        for i in range(min(num_relevant, k)):
            idcg += 1.0 / np.log2(i + 2)
        
        # Return NDCG
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def compute_embedding_reward(self, predicted_semantic: str, ground_truth: Union[str, List[str]]) -> float:
        """Compute embedding-based reward using NDCG@1000
        
        Args:
            predicted_semantic: Predicted semantic ID from LLM
            ground_truth: Ground truth semantic ID(s)
            
        Returns:
            NDCG@1000 score as reward
        """
        # Ensure ground truth is a list
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        
        return self.calculate_ndcg_at_k(predicted_semantic, ground_truth, k=1000)

# Global instance for efficient reuse
_embedding_calculator = None

def get_embedding_calculator() -> EmbeddingRewardCalculator:
    """Get global embedding calculator instance"""
    global _embedding_calculator
    if _embedding_calculator is None:
        _embedding_calculator = EmbeddingRewardCalculator()
    return _embedding_calculator

def compute_embedding_score(solution_str: str, ground_truth: Union[str, Dict, List[str]]) -> Dict:
    """Compute embedding-based semantic sequence recommendation score
    
    Args:
        solution_str: Raw model solution
        ground_truth: Ground truth answer or semantic ID string/dict/list
        
    Returns:
        Dictionary containing scoring results
    """
    # Extract predicted answer
    predicted_answer, processed_response = extract_solution(solution_str)
    
    # Parse ground_truth
    if isinstance(ground_truth, dict):
        if 'target_semantic_id' in ground_truth:
            true_semantic_ids = ground_truth['target_semantic_id']
        elif 'ground_truth' in ground_truth:
            true_semantic_ids = ground_truth['ground_truth']
        else:
            true_semantic_ids = str(ground_truth)
    else:
        true_semantic_ids = ground_truth
    
    # Ensure true_semantic_ids is a list
    if isinstance(true_semantic_ids, str):
        true_semantic_ids = [true_semantic_ids]
    
    # If no prediction extracted, return zero score
    if predicted_answer is None:
        return {
            'score': 0.0,
            'predicted': None,
            'ground_truth': true_semantic_ids,
            'ndcg_1000': 0.0,
            'error': 'Failed to extract prediction'
        }
    
    # Calculate NDCG@1000 using embeddings
    calculator = get_embedding_calculator()
    ndcg_score = calculator.compute_embedding_reward(predicted_answer, true_semantic_ids)
    
    return {
        'score': ndcg_score,
        'predicted': normalize_semantic_id(predicted_answer),
        'ground_truth': [normalize_semantic_id(gt) for gt in true_semantic_ids],
        'ndcg_1000': ndcg_score,
        'raw_prediction': predicted_answer
    }

# Main scoring function
def compute_sem_seq_rec_score(solution_str: str, ground_truth: Union[str, Dict]) -> float:
    """Main function to compute semantic sequence recommendation score
    
    Args:
        solution_str: Model solution
        ground_truth: Ground truth
        
    Returns:
        Score (0.0 - 1.0)
    """
    result = compute_score(solution_str, ground_truth)
    return result['score']

# New main scoring function using embeddings
def compute_sem_seq_rec_embedding_score(solution_str: str, ground_truth: Union[str, Dict, List[str]]) -> float:
    """Main function to compute embedding-based semantic sequence recommendation score
    
    Args:
        solution_str: Model solution
        ground_truth: Ground truth
        
    Returns:
        NDCG@1000 score (0.0 - 1.0)
    """
    result = compute_embedding_score(solution_str, ground_truth)
    return result['score']