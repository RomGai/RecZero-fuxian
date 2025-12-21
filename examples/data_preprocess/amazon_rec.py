""" Preprocess dataset for amazon recommendation task """

import os
from datasets import Dataset, load_dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
import pandas as pd
import random

def make_prefix_rec(dp, template_type):
    quiz = dp['quiz']
    candidates = dp['candidates']
    if template_type == 'base':
        prefix = f"""The user asks for product recommendations, and the Assistant helps to make recommendations. Based on the user's interaction history, the assistant first analyzes the user's preferences and requirements, then provides the user with the final recommendation from the provided list of candidate products. The analysis process and recommendation are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> analysis process here </think><answer> recommended product here</answer>. Now the user is seeking a product recommendation from you. After thinking, when you finally reach a conclusion, please provide only the corresponding recommended product name within the tags.\n\nUser:{quiz}\nAssistant: <think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\nYou are a helpful shopping assistant. Based on the user's interaction history, you first analyze the user's preferences and requirements, then provide the user with the final recommendation from the provided list of candidate products. The analysis process and recommendation are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> analysis process here </think><answer> recommended product here</answer>. Now the user is seeking a product recommendation from you. After thinking, when you finally reach a conclusion, please provide only the corresponding recommended product name within the tags.\n<|im_end|>\n<|im_start|>user\ninteraction history: {quiz}\n candidates: {candidates}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    return prefix

def save_candidates_parquet(dataset, local_dir, num_candidates):
    """Save dataset with a specific number of candidates to a Parquet file."""
    def sample_candidates(example):
        # Ensure the ground truth is always included
        ground_truth = example['solution_text_format']
        candidates = example['candidates']
        
        # Remove ground truth from candidates and sample
        candidates = [c for c in candidates if c != ground_truth]
        sampled_candidates = random.sample(candidates, num_candidates - 1)
        
        # Add ground truth back to the sampled candidates
        sampled_candidates.append(ground_truth)
        random.shuffle(sampled_candidates)  # Shuffle to randomize order
        
        example['candidates'] = sampled_candidates
        return example

    # Apply sampling to the dataset
    sampled_dataset = dataset.map(sample_candidates)
    
    # Save to Parquet
    filename = f'{num_candidates}candidates.parquet'
    sampled_dataset.to_parquet(os.path.join(local_dir, filename))
    print(f"Saved {num_candidates} candidates dataset to {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # to change
    parser.add_argument('--local_dir', default='/home/jiangjunguang.jjg/code_linlin/Logic-RL-main/data/amazon_beauty_candidates')
    parser.add_argument('--hdfs_dir', default=None)
    # to change
    parser.add_argument('--data_path', default='/home/jiangjunguang.jjg/code_linlin/Logic-RL/data/amazon_beauty_origin/Beauty_10candidates.jsonl')
    parser.add_argument('--train_size', type=int, default=4000)
    parser.add_argument('--test_size', type=int, default=100)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    
    args = parser.parse_args()
    
    data_source = 'amazon_rec'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    # Load custom JSONL dataset
    def gen_from_jsonl(path):
        with open(path) as f:
            for line in f:
                yield json.loads(line)
    
    raw_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': args.data_path})
    print(len(raw_dataset))

    assert len(raw_dataset) >= TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    # 处理额外的数据集
    extra_dataset = raw_dataset.select(range(TRAIN_SIZE + TEST_SIZE, len(raw_dataset)))

    # 将额外的数据集转换为 DataFrame
    extra_df = pd.DataFrame(extra_dataset)

    # 打印 DataFrame 的信息以进行调试
    print("Extra DataFrame length:", len(extra_df))
    print("Extra DataFrame columns:", extra_df.columns)
    print("Extra DataFrame head:", extra_df.head())

    # 保存 DataFrame 为 Parquet 文件
    extra_df.to_parquet(os.path.join(args.local_dir, 'extra.parquet'), index=False)
    print("保存额外数据集")

    # 新增：将包含 candidates 的 extra 数据保存为 JSON 文件
    extra_data_with_candidates_list = []
    for idx, row in extra_df.iterrows():
        extra_data_with_candidates = {
            "system": "You are a helpful shopping assistant.",
            "instruction": "Based on the user's interaction history and the list of candidate products, provide the user with the recommended product.",
            "input": f"Now the user is seeking a recommended product from you. interaction history: {row['quiz']}, candidates: {row['candidates']}",
            "output": row['solution_text_format'],
        }
        extra_data_with_candidates_list.append(extra_data_with_candidates)

    # 保存为新的 JSON 文件
    with open(os.path.join(args.local_dir, 'amazon_rec_with_candidates_demo.json'), 'w', encoding='utf-8') as f:
        json.dump(extra_data_with_candidates_list, f, ensure_ascii=False, indent=2)

    print("保存包含 candidates 的额外数据集为 JSON 文件")

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix_rec(example, template_type=args.template_type)
            solution = {
                "solution_text_format": example['solution_text_format'],
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "recommendation",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if not exists
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

    # Save datasets with different numbers of candidates
    for num_candidates in [3, 5, 7, 10]:
        save_candidates_parquet(train_dataset, local_dir, num_candidates)