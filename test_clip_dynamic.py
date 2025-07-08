import os
import clip
import json
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from peft import LoraConfig, get_peft_model

from config import *

def load_finetuned_clip(checkpoint_path, device, model_name):
    """加载微调后的CLIP模型"""
    # 加载原始CLIP模型
    model, preprocess = clip.load(model_name, device=device)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 如果使用了LoRA微调，重新应用LoRA配置
    if 'lora_config' in checkpoint and checkpoint['lora_config'] is not None:
        lora_config = checkpoint['lora_config']
        model = get_peft_model(model, lora_config).float().to(device)
    
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, preprocess

def get_class_names(data_dir):
    """从文件夹名称获取类别名称"""
    return [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

def predict_image(image_path, model, preprocess, class_names, label_dict, device):
    """预测单张图片的类别（三步渐进式推理版本）"""
    # 加载和预处理图片
    image = Image.open(image_path).convert('RGB')
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    # 准备双模态文本提示
    text_inputs_coarse = []  # 粗粒度提示
    text_inputs_fine = []    # 细粒度提示（每个提示只含一个属性）
    # 修改点1：分离粗/细粒度提示生成
    for coarse_label, info in label_dict.items():
        # 粗粒度模板："The terrain is {}"
        text_inputs_coarse.append(f"The terrain is {coarse_label}")
        
        # 细粒度模板："The terrain is {}, characterized by {}"（单属性）
        for attr in info['fine_labels']:
            text_inputs_fine.append(f"The terrain is {coarse_label}, characterized by {attr}")
    
    # 编码所有文本提示
    text_tokens_coarse = clip.tokenize(text_inputs_coarse).to(device)
    text_tokens_fine = clip.tokenize(text_inputs_fine).to(device)
    
    # 双模态特征提取
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features_coarse = model.encode_text(text_tokens_coarse)
        text_features_fine = model.encode_text(text_tokens_fine)
        
        # 特征归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_coarse = text_features_coarse / text_features_coarse.norm(dim=-1, keepdim=True)
        text_features_fine = text_features_fine / text_features_fine.norm(dim=-1, keepdim=True)
        
        # 计算相似度（原始logits）
        S_coarse = (100.0 * image_features @ text_features_coarse.T).softmax(dim=-1) # [1, C]
        S_fine = (100.0 * image_features @ text_features_fine.T).softmax(dim=-1) # [1, K*C]
        # S_coarse = image_features @ text_features_coarse.T  # [1, C]
        # S_fine = image_features @ text_features_fine.T      # [1, K*C]
    
    # 属性感知相似度融合
    coarse_similarity = {}
    fine_similarity = {}
    lambda_weight = 0.3  # 多样性权重
    
    # 修改点2：实现属性投票机制
    for c_idx, coarse_label in enumerate(label_dict.keys()):
        # 粗粒度分支
        coarse_score = S_coarse[0, c_idx].item()
        
        # 细粒度分支
        attr_mask = [i for i, text in enumerate(text_inputs_fine) if coarse_label in text]
        attr_scores = S_fine[0, attr_mask]
        
        # 计算聚合分数：均值 + λ*最大值
        # mean_score = attr_scores.mean().item()
        sum_score = attr_scores.sum().item()
        max_score = attr_scores.max().item()
        fine_score = sum_score + lambda_weight * max_score
        
        # 改用加权中位数抵抗离群点
        # sorted_scores = torch.sort(attr_scores).values
        # median_score = sorted_scores[len(sorted_scores)//2]
        # fine_score = 0.7*median_score + 0.3*mean_score
        
        coarse_similarity[coarse_label] = coarse_score
        fine_similarity[coarse_label] = fine_score
    
    # 修改点3：动态权重决策
    # 计算粗粒度分支的不确定性（熵）
    probs_coarse = torch.softmax(S_coarse / 0.07, dim=-1)
    entropy = -torch.sum(probs_coarse * torch.log(probs_coarse + 1e-9), dim=-1)
    # alpha = torch.sigmoid(2.5 * entropy).item()  # β=2.5
    
    # 计算粗粒度置信度：1 - 标准化熵
    # max_prob = probs_coarse.max().item()
    entropy_norm = entropy / np.log(len(class_names))  # 归一化到[0,1]
    # confidence = 0.5 * max_prob + 0.5 * (1 - entropy_norm)  # 综合最大概率和熵
    
    # 动态权重计算（置信度越低，细粒度权重越高）
    # alpha = 1 - confidence  # 直接映射
    alpha = entropy_norm
    
    print(alpha, coarse_similarity, fine_similarity)
    
    # 分数融合
    final_scores = {}
    for coarse_label in label_dict.keys():
        score = (1-alpha)*coarse_similarity[coarse_label] + alpha*fine_similarity[coarse_label]
        final_scores[coarse_label] = score
    
    # 获取预测结果
    predicted_coarse = max(final_scores, key=final_scores.get)
    
    # 获取Top3细粒度属性（仅用于分析）
    attr_mask = [i for i, text in enumerate(text_inputs_fine) if predicted_coarse in text]
    top3_attrs = []
    if attr_mask:
        fine_probs = torch.softmax(S_fine[0, attr_mask] * 100.0, dim=-1)
        top3_idx = fine_probs.topk(3).indices
        top3_attrs = [text_inputs_fine[attr_mask[idx]].split("characterized by ")[-1] for idx in top3_idx]
    
    # print(predicted_coarse, top3_attrs, final_scores[predicted_coarse], fine_probs.tolist())
    
    return predicted_coarse, top3_attrs, final_scores[predicted_coarse], fine_probs.tolist()


def evaluate_dataset(data_dir, model, preprocess, class_names, label_dict, device):
    """评估整个数据集"""
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    results = []
    
    # 为每个类别创建计数器
    class_stats = {c: {'top1_correct': 0, 'top3_correct': 0, 'total': 0} for c in class_names}
    
    # 遍历所有类别文件夹
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        
        # 遍历该类别下的所有图片
        for img_name in tqdm(os.listdir(class_dir), desc=f"Processing {class_name}"):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                predicted_coarse, predicted_fine, coarse_score, fine_scores = predict_image(img_path, model, preprocess, class_names, label_dict, device)
                
                # 获取预测结果
                is_correct_top1 = (predicted_coarse == class_name)
                is_correct_top3 = (class_name in [predicted_coarse] + predicted_fine)
                
                # 更新计数
                correct_top1 += int(is_correct_top1)
                correct_top3 += int(is_correct_top3)
                total += 1
                
                # 更新类别统计
                class_stats[class_name]['total'] += 1
                class_stats[class_name]['top1_correct'] += int(is_correct_top1)
                class_stats[class_name]['top3_correct'] += int(is_correct_top3)

                # 记录结果
                results.append({
                    'image_path': img_path,
                    'true_label': class_name,
                    'predicted_coarse': predicted_coarse,
                    'predicted_fine': predicted_fine,
                    'coarse_score': coarse_score.tolist(),
                    'fine_scores': fine_scores,
                    'correct_top1': is_correct_top1,
                    'correct_top3': is_correct_top3
                })
    
    # 计算总体准确率
    accuracy_top1 = correct_top1 / total
    accuracy_top3 = correct_top3 / total
    
    return accuracy_top1, accuracy_top3, results, class_stats


def main():
    global data_dir, model_name, fine_tune_method, save_dir, label_dict_path, save_test_file
    
    # 新增：解析命令行参数
    parser = argparse.ArgumentParser(description="Test CLIP with flexible parameters.")
    parser.add_argument("--data_dir", type=str, default=data_dir, help="Path to JSON dataset directory")
    parser.add_argument("--model_name", type=str, default=model_name, choices=["RN50", "ViT-B/32", "ViT-B/16", "ViT-L/14"], help="CLIP model name")
    parser.add_argument("--fine_tune_method", type=str, default=fine_tune_method, choices=["full", "lora"], help="Fine-tuning method")
    parser.add_argument("--save_dir", type=str, default=save_dir, help="Model save directory")
    parser.add_argument("--label_dict_path", type=str, default=label_dict_path, help="Path to fine-grained label dictionary")
    parser.add_argument("--save_test_file", type=str, default=save_test_file, help="Test results file name")
    args = parser.parse_args()

    # 覆盖 config 参数
    data_dir = args.data_dir
    model_name = args.model_name
    fine_tune_method = args.fine_tune_method
    save_dir = args.save_dir
    label_dict_path = args.label_dict_path
    save_test_file = args.save_test_file
    
    # 设置参数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 获取类别名称
    with open(label_dict_path, 'r') as f:
        label_dict = json.load(f)
    class_names = get_class_names(data_dir)
    print(f"Found {len(class_names)} classes: {class_names}")
    
    best_acc = 0
    
    for checkpoint_path in Path(save_dir).rglob("*.pt"):
        checkpoint_path = str(checkpoint_path)
        print(f"Loading model from: {checkpoint_path}")
    
        # 加载模型
        model, preprocess = load_finetuned_clip(checkpoint_path, device, model_name)
    
        # 评估数据集
        accuracy_top1, accuracy_top3, results, class_stats = evaluate_dataset(
            data_dir, model, preprocess, class_names, label_dict, device
        )
        # 打印总体结果
        print(f"\nOverall Accuracy:")
        print(f"Top-1: {accuracy_top1:.4f}")

        if accuracy_top1 > best_acc:
            best_acc = accuracy_top1
            
            # 保存每个类别的准确率
            class_accuracies = {}
            for class_name, stats in class_stats.items():
                top1_acc = stats['top1_correct'] / stats['total']
                class_accuracies[class_name] = {
                    'accuracy': top1_acc,
                    'correct': stats['top1_correct'],
                    'total': stats['total']
                }
                # print(f"{class_name}: {class_acc:.4f} ({stats['correct']}/{stats['total']})")
        
            # 保存详细结果
            with open(os.path.join(save_dir, save_test_file), 'w') as f:
                json.dump({
                    'accuracy': accuracy_top1,
                    'class_accuracies': class_accuracies,
                    'results': results
                }, f, indent=2)

if __name__ == "__main__":
    main()