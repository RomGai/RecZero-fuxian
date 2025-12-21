"""
Preprocess Sports dataset for Amazon LC-REC task
Converts item IDs to semantic IDs for user history while keeping target as item_id
"""

import os
import json
import pandas as pd
import numpy as np
from datasets import Dataset
from tqdm import tqdm
import argparse
from typing import Dict, List, Any


def load_sports_data(data_dir: str):
    """Load Sports dataset files"""
    print("Loading Sports dataset...")
    
    # Load index mapping (item_id -> semantic_id)
    with open(os.path.join(data_dir, 'Sports.index.json'), 'r') as f:
        index_mapping = json.load(f)
    
    # Load item metadata
    with open(os.path.join(data_dir, 'Sports.item.json'), 'r') as f:
        item_metadata = json.load(f)
    
    # Load interaction files
    train_inter = pd.read_csv(os.path.join(data_dir, 'Sports.train.inter'), sep='\t')
    test_inter = pd.read_csv(os.path.join(data_dir, 'Sports.test.inter'), sep='\t')
    valid_inter = pd.read_csv(os.path.join(data_dir, 'Sports.valid.inter'), sep='\t')
    
    return index_mapping, item_metadata, train_inter, test_inter, valid_inter


def item_ids_to_semantic_ids(item_ids: List[int], index_mapping: Dict[str, List[str]]) -> str:
    """Convert list of item IDs to semantic ID string"""
    semantic_ids = []
    for item_id in item_ids:
        semantic_tokens = index_mapping.get(str(item_id), [f"<unknown_{item_id}>"])
        # Join the semantic tokens for this item
        semantic_ids.append(" ".join(semantic_tokens))
    return ", ".join(semantic_ids)


def format_item_info(item_id: str, item_metadata: Dict[str, Any]) -> str:
    """Format item metadata into readable string"""
    if item_id not in item_metadata:
        return f"Item {item_id}: No metadata available"
    
    item = item_metadata[item_id]
    info_parts = []
    
    # Add title
    if 'title' in item:
        info_parts.append(f"Title: {item['title']}")
    
    # Add description (truncated)
    if 'description' in item:
        desc = str(item['description'])
        if len(desc) > 400:
            desc = desc[:400] + "..."
        info_parts.append(f"Description: {desc}")
    
    # Add brand
    if 'brand' in item:
        info_parts.append(f"Brand: {item['brand']}")
    
    return " | ".join(info_parts)


def create_prompt_template(inters: str) -> str:
    """Create the standardized prompt template"""
    return f"<|im_start|>system\nI find the user's historical interactive items {inters}, and I want to know what next item the user needs. Can you help me decide?\n<|im_end|>\n<|im_start|>assistant\n"


def process_interaction_data(inter_df: pd.DataFrame, index_mapping: Dict[str, List[str]], 
                           item_metadata: Dict[str, Any], split_name: str) -> List[Dict]:
    """Process interaction data into the required format"""
    processed_data = []
    
    # Calculate user and item averages (placeholder values for now)
    
    for idx, row in tqdm(inter_df.iterrows(), total=len(inter_df), desc=f"Processing {split_name}"):
        user_id = row['user_id:token']
        item_id_list = row['item_id_list:token_seq']
        target_item_id = row['item_id:token']

        # Parse item_id_list (space-separated string to list of integers)
        if pd.isna(item_id_list) or str(item_id_list).strip() == '':
            history_item_ids = []
        else:
            history_item_ids = [int(x) for x in str(item_id_list).split()]
        
        # Convert history item IDs to semantic IDs
        inters = item_ids_to_semantic_ids(history_item_ids, index_mapping)
        
        # Convert target item ID to semantic ID
        target_semantic_id = item_ids_to_semantic_ids([target_item_id], index_mapping)
        
        # Create prompt
        prompt_text = create_prompt_template(inters)
        
        # Format target item info
        target_item_info = format_item_info(str(target_item_id), item_metadata)
        
        # Create the data entry
        data_entry = {
            'user_history': inters,  # Semantic IDs for history
            'item_history': target_item_info,  # Target item metadata
            'ground_truth': int(target_item_id),  # Keep target as item_id
            'target_semantic_id': target_semantic_id,  # Target item semantic ID
            'data_source': 'sem_seq_rec',
            'prompt': [{'content': prompt_text, 'role': 'user'}],
            'ability': 'sem_seq_rec',
            'reward_model': {
                'ground_truth': {'ground_truth': str(target_item_id), 
                                 'target_semantic_id': str(target_semantic_id),
                                 },
                'style': 'rule'
            },
            'extra_info': {
                'index': idx,
                'split': split_name,
                'user_id': int(user_id),
                'history_length': len(history_item_ids),
                'target_item_id': int(target_item_id)
            }
        }
        
        processed_data.append(data_entry)
    
    return processed_data


def save_dataset(data: List[Dict], output_path: str, split_name: str):
    """Save processed data to parquet format"""
    dataset = Dataset.from_list(data)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save to parquet
    output_file = os.path.join(output_path, f'{split_name}.parquet')
    dataset.to_parquet(output_file)
    
    print(f"Saved {len(data)} samples to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Preprocess Sports dataset for Amazon LC-REC task')
    parser.add_argument('--data_path', default='/home/huangyanwen.hyw/code_linlin/RecZero/data/Sports', 
                       help='Path to Sports dataset directory')
    parser.add_argument('--local_dir', default='/home/huangyanwen.hyw/code_linlin/RecZero/data/sports_lcrec',
                       help='Output directory for processed data')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples per split (for testing, applies to all splits)')
    
    # Individual split length controls
    parser.add_argument('--max_train_samples', type=int, default=None,
                       help='Maximum number of samples for train split')
    parser.add_argument('--max_test_samples', type=int, default=2048,
                       help='Maximum number of samples for test split')
    parser.add_argument('--max_valid_samples', type=int, default=None,
                       help='Maximum number of samples for valid split')
    
    args = parser.parse_args()
    
    # Load data
    index_mapping, item_metadata, train_inter, test_inter, valid_inter = load_sports_data(args.data_path)
    
    print(f"Loaded {len(index_mapping)} items in index mapping")
    print(f"Loaded {len(item_metadata)} items in metadata")
    print(f"Train interactions: {len(train_inter)}")
    print(f"Test interactions: {len(test_inter)}")
    print(f"Valid interactions: {len(valid_inter)}")
    
    # Process each split with individual limits
    splits_config = [
        ('train', train_inter, args.max_train_samples),
        ('test', test_inter, args.max_test_samples),
        ('valid', valid_inter, args.max_valid_samples)
    ]
    
    for split_name, inter_df, max_split_samples in splits_config:
        print(f"\nProcessing {split_name} split...")
        
        # Apply split-specific limit first, then general limit
        current_limit = max_split_samples
        if current_limit is None:
            current_limit = args.max_samples
        
        if current_limit is not None and len(inter_df) > current_limit:
            inter_df = inter_df.head(current_limit)
            print(f"Limited {split_name} to {current_limit} samples")
        
        # Process the data
        processed_data = process_interaction_data(inter_df, index_mapping, item_metadata, split_name)
        
        # Save to parquet
        save_dataset(processed_data, args.local_dir, split_name)
    
    print(f"\nProcessing complete! Output saved to {args.local_dir}")
    
    # Print sample for verification
    if processed_data:
        print("\nSample processed entry:")
        sample = processed_data[0]
        for key, value in sample.items():
            if key == 'prompt':
                print(f"{key}: {value[0]['content'][:]}")
            elif isinstance(value, str):
                print(f"{key}: {value[:]}")
            else:
                print(f"{key}: {value}")


if __name__ == '__main__':
    main()