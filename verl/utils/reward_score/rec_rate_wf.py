import re
from typing import Dict, Tuple, Optional

def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str

def extract_tag_content(response: str, tag_name: str) -> str:
    """提取指定标签中的内容
    
    参数:
        response: 模型响应字符串
        tag_name: 标签名称（不含尖括号）
        
    返回:
        标签内的内容，如果未找到则返回空字符串
    """
    pattern = f'<{tag_name}>(.*?)</{tag_name}>'
    match = re.search(pattern, response, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return ""

def validate_response_structure(response: str) -> bool:
    """验证响应结构是否符合要求
    
    参数:
        response: 模型响应字符串
        
    返回:
        布尔值，表示格式是否正确
    """
    print("\n[结构验证]")
    validation_passed = True

    # 检查必需标签
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = response.count(tag_str)
        positions[tag_name] = pos = response.find(tag_str)
        
        print(f"  {tag_str}: 出现次数={count}, 位置={pos}")
        
        if count != expected_count:
            print(f"  [错误] {tag_str} 出现 {count} 次 (期望 {expected_count} 次)")
            validation_passed = False

    # 验证标签顺序
    expected_order = [
        ('think_start', 'think_end'),
        ('think_end', 'answer_start'),
        ('answer_start', 'answer_end')
    ]

    for prev_tag, next_tag in expected_order:
        if positions[prev_tag] > positions[next_tag]:
            print(f"  [错误] 标签顺序错误: {tags[prev_tag][0]} 应该在 {tags[next_tag][0]} 之前")
            validation_passed = False
            break
    else:
        print("  标签顺序验证通过")

    # 提取并验证内容
    if validation_passed:
        # 验证<think>内容
        think_content = extract_tag_content(response, "think")
        if not think_content:
            print("  [错误] <think>标签内容为空")
            validation_passed = False

        # 验证<answer>内容
        answer_content = extract_tag_content(response, "answer")
        if answer_content:
            try:
                rating = float(answer_content)
                if rating < 1 or rating > 10:
                    print(f"  [错误] 评分 {rating} 超出有效范围(1-10)")
                    validation_passed = False
            except ValueError:
                print(f"  [错误] 评分 '{answer_content}' 不是有效的数字")
                validation_passed = False
        else:
            print("  [错误] <answer>标签内容为空")
            validation_passed = False

    return validation_passed

def compute_score(solution_str: str, 
                 ground_truth: Dict[str, str],
                 format_reward: int = 1,
                 answer_reward: float = 1.0) :
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Total score (sum of format and answer rewards)
    """
    print("\n" + "="*80)
    print(" Processing New Sample ".center(80, '='))

    print("solution_str:", solution_str)
    
    # Parse ground truth data
    """solution_text = ground_truth.get('solution_text_format', '')
    gt_status = parse_solution_text_format(solution_text)
    expected_names = list(gt_status.keys())
    print(f"[Ground Truth] Final identities: {gt_status}")"""

    gt_status = ground_truth.get('ground_truth', '')
    # expected_names = list(gt_status.keys())
    print(f"[Ground Truth] Final item: {gt_status}")

    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Model Response]\n{processed_str}")

    # Validate response structure
    format_correct = validate_response_structure(processed_str)
    format_score = format_reward if format_correct else -abs(format_reward)
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")

    # Validate answer content
    answer_score = 0
    if format_correct and answer_text:
        # to change
        # pred_status = parse_model_answer(answer_text, expected_names)
        # pred_status = round(float(answer_text))
        pred_status = float(answer_text)
        if pred_status:
            print(f"\n[Content Validation]")
            print(f"  Expected: {gt_status}")
            print(f"  Predicted: {pred_status}")
            
            """if round(float(pred_status)) == round(float(gt_status)):
                answer_score = 2
                print("  Content validation: FULL MATCH")
            else:
                # answer_score = -1.5
                print("  Content validation: MISMATCH")
                answer_score = 1/abs(round(float(pred_status)) - round(float(gt_status)))"""
            answer_score = 2 - abs(float(pred_status) - float(gt_status))/2
        else:
            answer_score = -2
            print( "Fail to parse answer")
    else:
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    total_score = format_score + answer_score
    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {answer_score}")
    print(f"  Total: {total_score}")
    print("="*80 + "\n")

    """try:
        pred_rating = float(answer_text) if answer_text else 0.0
    except (ValueError, TypeError):
        pred_rating =0.0"""
        
    try:
        gt_rating = float(gt_status)
        try:
            pred_rating = float(answer_text) if answer_text else gt_rating
        except (ValueError, TypeError):
            pred_rating = gt_rating
    except (ValueError, TypeError):
        gt_rating = 0.0
        pred_rating = 0.0

    # 修改返回值，同时返回预测值和真实值
    return {
        'score': total_score,
        'pred_rating': pred_rating,
        'gt_rating': gt_rating
    }