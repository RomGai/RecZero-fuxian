""" Preprocess dataset for amazon recommendation task """

import os
from datasets import Dataset, load_dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json

def make_prefix_rec(dp, template_type):
    history = dp['user_history']
    final_item_meta = dp['item_history']
    # user_avg = dp['user_avg']
    # item_avg = dp['item_avg']
    if template_type == 'base':
        prefix = f"""The user asks for product recommendations, and the Assistant helps to make recommendations. Based on the user's interaction history, the assistant first analyzes the user's preferences and requirements, then provides the user with the final recommendation from the provided list of candidate products. The analysis process and recommendation are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> analysis process here </think><answer> recommended product here</answer>. Now the user is seeking a product recommendation from you. After thinking, when you finally reach a conclusion, please provide only the corresponding recommended product name within the tags.\n\nUser:{quiz}\nAssistant: <think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\nYou are a helpful shopping assistant. Based on the user's purchase history, You first think, then predict if the user will like the target item. The answer 1 means yes, and the answer 0 means no. The analysis process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> analysis process here </think><answer> 0 or 1 </answer>. After thinking, when you finally reach a conclusion, please provide only the answer 0 or 1 within <answer> </answer> tags. i.e., <answer> 1 </answer>. Now if they will like the target item? \n<|im_end|>\n<|im_start|>user\n
        purchase history: {history}\ntarget item: {final_item_meta}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    return prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # to change
    parser.add_argument('--local_dir', default='/home/huangyanwen.hyw/code_linlin/Logic-RL-rating/data/reason4rec_book_ctr_test')
    parser.add_argument('--hdfs_dir', default=None)
    # to change
    parser.add_argument('--data_path', default='/home/huangyanwen.hyw/code_linlin/Logic-RL-rating/data/Reason4Rec/Music_data/raw_features_data_ctr_collm_test.jsonl')
    parser.add_argument('--train_size', type=int, default=1)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    
    args = parser.parse_args()
    
    data_source = 'amazon_ctr'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    # Load custom JSONL dataset
    def gen_from_jsonl(path):
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                # 统一转成字符串，防止 ArrowTypeError
                for k in ("user_history", "item_history"):
                    v = data.get(k)
                    if isinstance(v, list):
                        # 你喜欢什么格式都行，这里用 " | " 连接
                        data[k] = " | ".join(map(str, v))
                yield data
    
    raw_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': args.data_path})
    print(len(raw_dataset))

    assert len(raw_dataset) >= TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))


    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix_rec(example, template_type=args.template_type)
            ground_truth = {
                "ground_truth": str(int(float(example['label']))),
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "rec_rating",
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