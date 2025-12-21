import re
from typing import Dict, Tuple, Optional

def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """从模型响应中提取最终答案
    
    Args:
        solution_str: 模型的原始响应字符串
        
    Returns:
        包含(提取的答案, 处理后的字符串)的元组
    """
    # 分离响应以隔离助手输出
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[错误] 未能找到模型响应头部")
        return None, solution_str

    # 使用XML风格标签提取最终答案
    rate_pattern = r'<rate>(.*?)</rate>'
    matches = list(re.finditer(rate_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[错误] 未找到有效的rate标签")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str

def extract_tag_content(response: str, tag_name: str) -> str:
    """提取指定标签中的内容
    
    Args:
        response: 模型响应字符串
        tag_name: 标签名称（不含尖括号）
        
    Returns:
        标签内的内容，如果未找到则返回空字符串
    """
    pattern = f'<{tag_name}>(.*?)</{tag_name}>'
    match = re.search(pattern, response, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return ""

def validate_response_structure(response: str) -> bool:
    """验证响应结构是否符合<analyze user><analyze item><match><rate>格式要求
    
    Args:
        response: 模型响应字符串
        
    Returns:
        布尔值，表示格式是否正确
    """
    print("\n[结构验证]")
    validation_passed = True

    # 检查必需标签
    required_tags = ['analyze user', 'analyze item', 'match', 'rate']
    
    for tag in required_tags:
        start_tag = f'<{tag}>'
        end_tag = f'</{tag}>'
        
        start_count = response.count(start_tag)
        end_count = response.count(end_tag)
        
        print(f"  {start_tag}: 出现次数={start_count}")
        print(f"  {end_tag}: 出现次数={end_count}")
        
        if start_count != 1 or end_count != 1:
            print(f"  [错误] {tag}标签缺失或重复")
            validation_passed = False

    # 验证标签顺序
    if validation_passed:
        tag_positions = []
        for tag in required_tags:
            start_pos = response.find(f'<{tag}>')
            end_pos = response.find(f'</{tag}>')
            
            if start_pos == -1 or end_pos == -1:
                print(f"  [错误] {tag}标签未找到")
                validation_passed = False
                break
                
            tag_positions.append((start_pos, end_pos, tag))
        
        # 检查标签是否按正确顺序出现
        tag_positions.sort()
        expected_order = ['analyze user', 'analyze item', 'match', 'rate']
        
        for i, (start_pos, end_pos, tag) in enumerate(tag_positions):
            if tag != expected_order[i]:
                print(f"  [错误] 标签顺序错误，期望{expected_order[i]}，实际{tag}")
                validation_passed = False
                break

    # 验证内容完整性
    if validation_passed:
        analyze_user_content = extract_tag_content(response, "analyze user")
        analyze_item_content = extract_tag_content(response, "analyze item")
        match_content = extract_tag_content(response, "match")
        rate_content = extract_tag_content(response, "rate")

        # 验证<analyze user>内容
        if not analyze_user_content:
            print("  [错误] <analyze user>标签内容为空")
            validation_passed = False

        # 验证<analyze item>内容
        if not analyze_item_content:
            print("  [错误] <analyze item>标签内容为空")
            validation_passed = False

        # 验证<match>内容
        if not match_content:
            print("  [错误] <match>标签内容为空")
            validation_passed = False

        # 验证<rate>内容
        if rate_content:
            try:
                rating = float(rate_content)
                if rating < 1 or rating > 5:
                    print(f"  [错误] 评分 {rating} 超出有效范围(1-5)")
                    validation_passed = False
            except ValueError:
                print(f"  [错误] 评分 '{rate_content}' 不是有效的数字")
                validation_passed = False
        else:
            print("  [错误] <rate>标签内容为空")
            validation_passed = False

    if validation_passed:
        print("  格式验证通过")

    return validation_passed

def compute_score(solution_str: str, 
                 ground_truth: Dict[str, str],
                 format_reward: int = 1,
                 answer_reward: float = 1.0):
    """计算模型响应的综合评分
    
    Args:
        solution_str: 模型的原始响应字符串
        ground_truth: 包含真实值数据的字典
        format_reward: 格式正确性奖励/惩罚分数
        answer_reward: 答案正确性奖励/惩罚分数
        
    Returns:
        包含评分信息的字典
    """
    print("\n" + "="*80)
    print(" 处理新样本 ".center(80, '='))

    print("solution_str:", solution_str)
    
    # 解析真实值数据
    gt_rating = ground_truth.get('ground_truth', '')
    print(f"[真实值] 评分: {gt_rating}")

    # 提取模型答案
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[模型响应]\n{processed_str}")

    # 验证响应结构
    format_correct = validate_response_structure(processed_str)
    format_score = format_reward if format_correct else -abs(format_reward)
    print(f"\n  格式验证: {'通过' if format_correct else '失败'}")
    print(f"  格式分数: {format_score}")

    # 验证答案内容
    answer_score = 0
    if format_correct and answer_text:
        pred_status = float(answer_text)
        gt_rating_float = float(gt_rating)
        if pred_status:            
            print(f"\n[内容验证]")
            print(f"  期望评分: {gt_rating_float}")
            print(f"  预测评分: {pred_status}")
            
            answer_score = 1 - abs(float(pred_status) - float(gt_rating_float))/2
        else:
            answer_score = -2
            print( "Fail to parse answer")
    else:
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    total_score = format_score + answer_score
    print("\n" + "-"*80)
    print(f" 最终评分 ".center(80, '-'))
    print(f"  格式分数: {format_score}")
    print(f"  答案分数: {answer_score}")
    print(f"  总分: {total_score}")
    print("="*80 + "\n")

    # 处理预测值和真实值
    try:
        gt_rating_float = float(gt_rating)
        try:
            pred_rating_float = float(answer_text) if answer_text else gt_rating_float
        except (ValueError, TypeError):
            pred_rating_float = gt_rating_float
    except (ValueError, TypeError):
        gt_rating_float = 0.0
        pred_rating_float = 0.0

    return {
        'score': total_score,
        'pred_rating': pred_rating_float,
        'gt_rating': gt_rating_float
    }