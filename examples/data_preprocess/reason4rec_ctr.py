"""
使用原始特征构建训练数据，不依赖 summarizer
适配原有的代码结构，直接替换 aspect_preference_summary 为原始特征组合
"""

import pandas as pd
from tqdm import tqdm
import json

def format_item_raw_features(item_row):
    """
    格式化单个 item 的原始特征，返回结构化文本
    """
    features = []
    
    # Title
    if pd.notna(item_row.get('title')):
        features.append(f"Title: {item_row['title']}")
    
    # Categories (如果存在)
    categories = item_row.get('categories')
    if categories is not None and not (isinstance(categories, float) and pd.isna(categories)):
        features.append(f"Categories: {categories}")
    
    # Brand (如果存在)  
    if pd.notna(item_row.get('brand')):
        features.append(f"Brand: {item_row['brand']}")
    
    # Price (如果存在)
    if pd.notna(item_row.get('price')):
        features.append(f"Price: {item_row['price']}")
    
    # Summary (如果存在，限制长度)
    if pd.notna(item_row.get('summary')):
        summary = str(item_row['summary'])
        if len(summary) > 200:
            summary = summary[:200] + "..."
        features.append(f"Summary: {summary}")
    
    # Description (如果存在，限制长度)
    if pd.notna(item_row.get('description')):
        desc = str(item_row['description'])
        if len(desc) > 400:
            desc = desc[:400] + "..."
        features.append(f"Description: {desc}")
    
    # Helpful votes (如果存在)
    # if pd.notna(item_row.get('helpful')):
    #     features.append(f"Helpful: {item_row['helpful']}")
    
    # Review text (限制长度)
    if pd.notna(item_row.get('reviews')):
        review = str(item_row['reviews'])
        if len(review) > 400:
            review = review[:400] + "..."
        features.append(f"Review: {review}")
    
    return " | ".join(features)

def format_item_fixed_features(item_row):
    """
    格式化商品的固定特征（title, categories, brand, price, description）
    """
    features = []
    
    # Title
    if pd.notna(item_row.get('title')):
        features.append(f"Title: {item_row['title']}")
    
    # Categories (如果存在)
    categories = item_row.get('categories')
    if categories is not None and not (isinstance(categories, float) and pd.isna(categories)):
        features.append(f"Categories: {categories}")
    
    # Brand (如果存在)  
    if pd.notna(item_row.get('brand')):
        features.append(f"Brand: {item_row['brand']}")
    
    # Price (如果存在)
    if pd.notna(item_row.get('price')):
        features.append(f"Price: {item_row['price']}")
    
    # Description (如果存在，限制长度)
    if pd.notna(item_row.get('description')):
        desc = str(item_row['description'])
        if len(desc) > 400:
            desc = desc[:400] + "..."
        features.append(f"Description: {desc}")
    
    return " | ".join(features)

def format_aggregated_item_history(item_history_data):
    """
    对同一商品的历史记录进行聚合，固定特征只显示一次，变化特征放入数组
    """
    if len(item_history_data) == 0:
        return ""
    
    # 获取第一条记录的固定特征
    first_record = item_history_data.iloc[0]
    fixed_features = format_item_fixed_features(first_record)
    
    # 收集所有变化的特征
    reviews = []
    summaries = []
    # helpful_votes = []
    ratings = []
    
    for _, record in item_history_data.iterrows():
        # 评分
        ratings.append(f"{float(record['ratings']):.1f}")
        
        # Review
        if pd.notna(record.get('reviews')):
            review = str(record['reviews'])
            if len(review) > 400:
                review = review[:400] + "..."
            reviews.append(review)
        
        # Summary
        if pd.notna(record.get('summary')):
            summary = str(record['summary'])
            if len(summary) > 200:
                summary = summary[:200] + "..."
            summaries.append(summary)
        
        # Helpful
        # if pd.notna(record.get('helpful')):
        #     helpful_votes.append(str(record['helpful']))
    
    # 构建聚合文本
    result = fixed_features
    
    # if ratings:
    #     result += f" | Ratings: [{', '.join(ratings)}]"
    
    if reviews:
        result += f" | Reviews: [{'; '.join(reviews)}]"
    
    if summaries:
        result += f" | Summaries: [{'; '.join(summaries)}]"
    
    # if helpful_votes:
    #     result += f" | Helpful: [{', '.join(helpful_votes)}]"
    
    return result

# 主处理逻辑
dataset = 'Music_data'
# product_class = 'Digital Music'

# 读取原始数据
data_df = pd.read_pickle(f'/home/huangyanwen.hyw/code_linlin/Logic-RL-rating/data/Reason4Rec/{dataset}/raw_data/test.pkl')

# 直接使用原始数据作为历史数据源
history_df = data_df.copy()

print(f"Available columns in data: {data_df.columns.tolist()}")

train_data = []
for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
    user_id = row['user_id']
    item_id = row['item_id']
    target_title = row['title']
    target_rating = row['ratings']
    
    user_history = history_df[history_df['user_id'] == user_id]
    # item_history = item_history[item_history['unixReviewTime'] < row['unixReviewTime']]
    """user_history = user_history[user_history['date'] < row['date']]
    user_history = user_history.sort_values(by='date', ascending=True)"""
    user_history = user_history[user_history['unixReviewTime'] < row['unixReviewTime']]
    user_history = user_history.sort_values(by='unixReviewTime', ascending=True)
    user_ctr = (user_history['ratings'] >= 3).mean()
    user_history = user_history.tail(10)

    item_history = history_df[history_df['item_id'] == item_id]
    """item_history = item_history[item_history['date'] < row['date']]
    item_history = item_history.sort_values(by='date', ascending=True)"""
    item_history = item_history[item_history['unixReviewTime'] < row['unixReviewTime']]
    item_history = item_history.sort_values(by='unixReviewTime', ascending=True)
    item_ctr = (item_history['ratings'] >= 3).mean()
    item_history = item_history.tail(10)

    if len(user_history) == 0 or len(item_history) == 0:
        continue
    data_df.at[idx, 'his_len'] = len(user_history) + len(item_history)

    # 构建用户历史文本 - 使用原始特征替代 aspect_preference_summary
    user_history_text = ''
    for i, (_, his) in enumerate(user_history.iterrows()):
        his_title = his['title']
        his_rating = his['ratings']
        # user_history_text += f"{i + 1}. {his_title}, {float(his_rating):.1f};\n"
        user_history_text += f"{i + 1}. {his_title}\n"
        # 替换 aspect_preference_summary 为原始特征组合
        raw_features = format_item_raw_features(his)
        user_history_text += f"{raw_features}\n\n"
    user_history_text = user_history_text.strip()
    
    # 构建商品历史文本 - 使用聚合格式
    item_history_text = format_aggregated_item_history(item_history)

    # 存储需要的字段，包含统计信息
    data_df.at[idx, 'user_history'] = user_history_text
    data_df.at[idx, 'item_history'] = item_history_text
    data_df.at[idx, 'ground_truth'] = 1 if target_rating >= 3 else 0
    data_df.at[idx, 'user_avg'] = user_ctr
    data_df.at[idx, 'item_avg'] = item_ctr

# 只保留有完整历史数据的样本
data_df = data_df.dropna(subset=['user_history', 'item_history'])

# 过滤和采样逻辑保持不变
data_df_sample = data_df[data_df['his_len'] >= 5].reset_index(drop=True)
# data_df_sample = data_df.sample(1000, random_state=0).reset_index(drop=True)

# 额外保存 JSONL 格式 - 包含统计信息
jsonl_data = []
for _, row in data_df_sample.iterrows():
    entry = {
        "user_history": row['user_history'],
        "item_history": row['item_history'], 
        "ground_truth": int(row['ground_truth']),
        "user_avg": float(row['user_avg']) if pd.notna(row['user_avg']) else 0.0,
        "item_avg": float(row['item_avg']) if pd.notna(row['item_avg']) else 0.0
    }
    jsonl_data.append(entry)

# 保存为 JSONL 文件
with open(f'/home/huangyanwen.hyw/code_linlin/Logic-RL-rating/data/Reason4Rec/{dataset}/raw_features_data_ctr_test.jsonl', 'w', encoding='utf-8') as f:
    for entry in jsonl_data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Generated {len(data_df_sample)} samples using raw features")
print(f"Saved JSONL to: /home/huangyanwen.hyw/code_linlin/Logic-RL-rating/data/Reason4Rec/{dataset}/raw_features_data_ctr_test.jsonl")

# 打印示例以验证格式
if len(jsonl_data) > 0:
    print("\n=== Example Entry ===")
    example = jsonl_data[0]
    print(f"Ground Truth (CTR): {example['ground_truth']}")
    print(f"User CTR: {example['user_avg']:.2f}")
    print(f"Item CTR: {example['item_avg']:.2f}")
    print("\n--- User History Preview (first 200 chars) ---")
    print(example['user_history'][:200] + "..." if len(example['user_history']) > 200 else example['user_history'])
    print("\n--- Item History Preview (first 200 chars) ---")
    print(example['item_history'][:200] + "..." if len(example['item_history']) > 200 else example['item_history'])
