""" Preprocess dataset for amazon recommendation task """

import os
from datasets import Dataset, load_dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json

def make_prefix_rec(dp, template_type):
    # history = dp['history']
    # final_item_meta = dp['final_item_meta']
    history = dp['user_history']
    final_item_meta = dp['item_history']
    user_avg = dp['user_avg']
    item_avg = dp['item_avg']
    if template_type == 'base':
        prefix = f"""The user asks for product recommendations, and the Assistant helps to make recommendations. Based on the user's interaction history, the assistant first analyzes the user's preferences and requirements, then provides the user with the final recommendation from the provided list of candidate products. The analysis process and recommendation are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> analysis process here </think><answer> recommended product here</answer>. Now the user is seeking a product recommendation from you. After thinking, when you finally reach a conclusion, please provide only the corresponding recommended product name within the tags.\n\nUser:{quiz}\nAssistant: <think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""You are a helpful assistant. Your task is to analyze a user's purchase history, summarize their preferences, analyze the given target item, and then analyze how well the given target item aligns with the user's preferences and predict a rating for that target item. \nPlease follow these steps precisely:

    1.  **Extract User Interest From Purchase History:**
        Based on the provided user_purchase_history, consolidate the user's overall preferences. Adhere strictly to the following format:
        ```
        <analyze user>
        User Preference: 
        ...
        </analyze user>\n
        ```
    2.  **Summarize The Given Target Item's Key Aspects:** Analyze the features and metadata of the target item provided in `target_item`. Predict potential points the user might like (`[pos]`) or dislike (`[neg]`) about this specific item, considering general product attributes. Adhere strictly to the following format:
        ```
        <analyze item>
        Product Name
        [pos] Summary of positive points
        [neg] Summary of negative points
        </analyze item>\n
        ```
    3.  **Evaluate Compatibility:** Internally reason (`<match>`) about *why* the target item would (or would not) be a good recommendation for this user. This reasoning should explicitly connect the user's overall preferences (from Step 1) with the target item's potential positive/negative aspects (from Step 2). Provide detailed justifications.
        ```
        <match>
        Reasoning here
        </match>\n
        ```
    4.  **Predict Rating:**
        Provide the final predicted rating for the target item within the `<rate>` tags. The answer should *only* contain the predicted numerical rating (e.g., on a 1-5 scale).
        Adhere strictly to the following format:
        ```
        <rate>
        Predicted rating for the target item
        </rate>\n
        ```
    Your inputs will be `user_history`,  `target_item`, `user_avg_rating` and `target_item_avg_rating`. Ensure your output follows the specified formats and uses the `<analyze user>`, `<analyze item>`, `<match>` and `<rate>` tags correctly.
    <|im_end|>
    <|im_start|>user
    user_history: {history}\n
    user_avg_rating: {user_avg}\n
    target_item: {final_item_meta}\n
    target_item_avg_rating: {item_avg}\n
    <|im_end|>
    <|im_start|>assistant
    """
    return prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # to change
    parser.add_argument('--local_dir', default='/home/huangyanwen.hyw/code_linlin/Logic-RL-rating/data/reason4rec_music_user_teacher')
    parser.add_argument('--hdfs_dir', default=None)
    # to change
    parser.add_argument('--data_path', default='/home/huangyanwen.hyw/code_linlin/Logic-RL-rating/data/Reason4Rec/Music_data/raw_features_data_teacher.jsonl')
    parser.add_argument('--train_size', type=int, default=16000)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    
    args = parser.parse_args()
    
    data_source = 'rec_rate_format'
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

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix_rec(example, template_type=args.template_type)
            ground_truth = {
                "ground_truth": str(int(float(example['ground_truth']))),
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "rec_rate",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth
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