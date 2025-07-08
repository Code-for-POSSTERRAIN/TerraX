import os
import clip
import json
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import confusion_matrix
import pandas as pd
import random

from config import *

class CustomCLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        # CLIP视觉编码器（冻结参数）
        self.vision_encoder = clip_model.visual
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # 获取CLIP视觉特征的维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(next(self.vision_encoder.parameters()).device)
            dummy_output = self.vision_encoder(dummy_input)
            feature_dim = dummy_output.shape[-1]
        
        # 三层MLP分类层
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, images):
        with torch.no_grad():
            features = self.vision_encoder(images)
        return self.classifier(features)

class ImageDataset(Dataset):
    def __init__(self, data_dir, preprocess, sample_ratio=1.0):
        self.class_names = [d for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d))]
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        
        self.images = []
        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append((
                        os.path.join(class_dir, img_name),
                        self.class_to_idx[class_name]
                    ))
        
        # 抽样
        if sample_ratio < 1.0:
            self.images = random.sample(self.images, int(len(self.images) * sample_ratio))
        
        self.preprocess = preprocess

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        return self.preprocess(image), label

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
    model.eval().to(device)
    
    return model, preprocess

def train_linear_layer(model, train_loader, test_loader, device, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    
    best_acc = 0.0
    best_model = None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # 验证
        test_acc, _, _ = evaluate_linear(model, test_loader, device)
        print(f"Epoch {epoch+1} | Loss: {running_loss:.2f} | test Acc: {test_acc:.2f}")
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = model
    
    return best_acc, best_model

def evaluate_linear(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    return correct / total, all_labels, all_preds

def save_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(save_path)

def main():
    # 直接在程序中指定参数
    train_dir = "/home4/TerraX-xinhai/new_dataset/JACKAL/train"
    test_dir = "/home4/TerraX-xinhai/new_dataset/JACKAL/test"
    model_name = "ViT-B/32"  
    # checkpoint_path = "/home4/TerraX/ckp/ViT_lora_our_data_new_dual_0.6_aug/clip_terrain_epoch_20.pt"  # 微调后的CLIP模型检查点路径
    # save_dir = "/home4/TerraX/ckp/ViT_lora_our_data_new_dual_0.6_aug"  # 结果保存路径
    checkpoint_name = "clip_terrain_epoch_20.pt"  # 微调后的CLIP模型检查点路径
    save_dir = "/home4/TerraX/ckp/ViT_lora_RSCD_4_dual_0.6_aug"  # 结果保存路径
    save_test_file = "few_shot_JACKAL_results.csv"  # 测试结果文件名
    num_epochs = 50  # 训练轮次
    batch_size = 32  # 批大小
    
    # 新增：解析命令行参数
    parser = argparse.ArgumentParser(description="Test CLIP with flexible parameters.")
    parser.add_argument("--train_dir", type=str, default=train_dir, help="Path to fine tune dataset directory")
    parser.add_argument("--test_dir", type=str, default=test_dir, help="Path to test dataset directory")
    parser.add_argument("--model_name", type=str, default=model_name, choices=["RN50", "ViT-B/32", "ViT-B/16", "ViT-L/14"], help="CLIP model name")
    parser.add_argument("--checkpoint_name", type=str, help="Test results file name")
    parser.add_argument("--save_dir", type=str, default=save_dir, help="Model save directory")
    parser.add_argument("--save_test_file", type=str, default=save_test_file, help="Test results file name")
    args = parser.parse_args()

    # 覆盖 config 参数
    train_dir = args.train_dir
    test_dir = args.test_dir
    model_name = args.model_name
    checkpoint_name = args.checkpoint_name
    save_dir = args.save_dir
    save_test_file = args.save_test_file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载微调后的CLIP模型
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    model, preprocess = load_finetuned_clip(checkpoint_path, device, model_name)
    
    # 创建自定义模型
    train_dataset = ImageDataset(train_dir, preprocess)
    test_dataset = ImageDataset(test_dir, preprocess)
    class_names = train_dataset.class_names
    custom_model = CustomCLIPClassifier(model, len(class_names)).to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 训练线性层
    acc, best_model = train_linear_layer(custom_model, train_loader, test_loader, device, num_epochs)
    
    # 评估最佳模型并保存混淆矩阵
    accuracy, all_labels, all_preds = evaluate_linear(best_model, test_loader, device)
    # save_confusion_matrix(all_labels, all_preds, class_names, os.path.join(save_dir, save_test_file))
    with open(os.path.join(save_dir, save_test_file), 'w') as f:
        json.dump({
            'accuracy': accuracy,
        }, f, indent=2)



if __name__ == "__main__":
    main()