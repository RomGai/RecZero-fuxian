import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl.utils.reward_score.kk import extract_solution, compute_score, parse_solution_text_format

def load_model_and_tokenizer(model_path):
    """加载模型和分词器"""
    print(f"正在加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=1024):
    """使用模型生成回答"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # 移除输入提示部分，只保留模型生成的内容
    response = response[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False)):]
    return response

def process_dataset(model_path, data_path, output_path):
    """处理数据集并保存结果"""
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # 加载数据集
    print(f"正在加载数据集: {data_path}")
    df = pd.read_parquet(data_path)
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理样本"):
        # 构建提示
        prompt = row['prompt']
        
        # 生成回答
        response = generate_response(model, tokenizer, prompt)
        
        # 提取答案
        answer_text, processed_str = extract_solution(response)
        
        # 解析真实答案
        ground_truth = {
            'solution_text_format': row['solution_text_format']
        }
        gt_status = parse_solution_text_format(row['solution_text_format'])
        
        # 计算得分
        score = compute_score(response, ground_truth)
        
        # 保存结果
        result = {
            'id': idx,
            'prompt': prompt,
            'full_response': response,
            'extracted_answer': answer_text,
            'processed_response': processed_str,
            'ground_truth': gt_status,
            'score': score
        }
        results.append(result)
        
        # 每10个样本保存一次结果，防止中断丢失数据
        if (idx + 1) % 10 == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 最终保存结果
    print(f"正在保存结果到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 计算总体统计信息
    total_samples = len(results)
    format_correct = sum(1 for r in results if r['score'] > -1)  # 格式正确的样本
    answer_correct = sum(1 for r in results if r['score'] > 1)   # 答案正确的样本
    
    print(f"\n总样本数: {total_samples}")
    print(f"格式正确样本数: {format_correct} ({format_correct/total_samples*100:.2f}%)")
    print(f"答案正确样本数: {answer_correct} ({answer_correct/total_samples*100:.2f}%)")
    
    return results

if __name__ == "__main__":
    model_path = "/home/jiangjunguang.jjg/LLM_models/qwq-32b"
    data_path = "/home/jiangjunguang.jjg/code_linlin/Logic-RL-main/data/kk/instruct/3ppl/test.parquet"
    output_path = "/home/jiangjunguang.jjg/code_linlin/Logic-RL-main/get_qwq_data/inference_results.json"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 处理数据集
    results = process_dataset(model_path, data_path, output_path)
    
    print(f"推理完成，结果已保存至 {output_path}")