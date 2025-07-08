import os

# # 指定使用第2块显卡
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import clip
import json
import wandb
import torch
import argparse
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torch.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from torch.cuda.amp import GradScaler, autocast
from peft import LoraConfig, get_peft_model

from config import *

class TerrainDataset(Dataset):
    def __init__(self, json_dir, transform=None, max_length=77):
        self.transform = transform
        self.max_length = max_length
        self.samples = []
        
        # 加载所有JSON文件
        for json_file in os.listdir(json_dir):
            if not json_file.endswith('.json'):
                continue
            with open(os.path.join(json_dir, json_file), 'r') as f:
                data = json.load(f)
                for sample in data:
                    self.samples.append(sample)
                    for aug_type in aug_types:
                        aug_sample = sample.copy()
                        # aug_sample['image'] = os.path.join(aug_dir, aug_type, os.path.basename(sample['image']))
                        # self.samples.append(aug_sample)
                        
                        image_path = sample['image']
                        base_name = os.path.basename(image_path)
                        two_level_dir = os.path.join(os.path.basename(os.path.dirname(image_path)), base_name)
                        aug_sample['image'] = os.path.join(aug_dir, aug_type, two_level_dir)
                        
                        if os.path.isfile(aug_sample['image']):
                            aug_sample['fine_grained_annotation'] = aug_sample['fine_grained_annotation'][:-1] + f", {aug_type}."
                            self.samples.append(aug_sample)
                
                # if isinstance(data, dict):  # 处理单个样本的情况
                #     self.samples.append(data)
                # else:  # 处理多个样本的情况
                #     self.samples.extend(data)
                        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载并预处理图像
        image = Image.open(sample['image']).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # 获取文本描述和标签
        coarse_text = sample['coarse_grained_annotation'][:self.max_length]
        fine_text = sample['fine_grained_annotation'][:self.max_length]
        label = sample['label']
        
        return image, coarse_text, fine_text, label

def train_clip(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, coarse_texts, fine_texts, labels) in enumerate(progress_bar):
        images = images.to(device)
        coarse_tokens = clip.tokenize(coarse_texts).to(device)
        fine_tokens = clip.tokenize(fine_texts).to(device)
        
        # 使用amp进行混合精度训练
        with autocast('cuda'):
            # 获取特征
            image_features = model.encode_image(images)
            coarse_features = model.encode_text(coarse_tokens)
            fine_features = model.encode_text(fine_tokens)
            
            # 特征归一化
            image_features = F.normalize(image_features, dim=-1)
            coarse_features = F.normalize(coarse_features, dim=-1)
            fine_features = F.normalize(fine_features, dim=-1)

            # 计算粗粒度损失
            logits_per_image = (image_features @ coarse_features.T) / temperature
            logits_per_text = logits_per_image.t()
            
            labels = torch.arange(len(images)).to(device)
            loss_coarse = (F.cross_entropy(logits_per_image, labels) 
                         + F.cross_entropy(logits_per_text, labels)) / 2
            
            # 计算细粒度损失
            fine_logits_per_image = (image_features @ fine_features.T) / temperature
            fine_logits_per_text = fine_logits_per_image.t()
            loss_fine = (F.cross_entropy(fine_logits_per_image, labels) 
                       + F.cross_entropy(fine_logits_per_text, labels)) / 2
            
            loss = coarse_weight * loss_coarse + fine_weight * loss_fine
           
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        wandb.log({
            "epoch": epoch,
            "loss": loss.item(),
            "loss_coarse": loss_coarse.item(),
            "loss_fine": loss_fine.item(),
            "image_text_similarity": logits_per_image.mean().item(),
        })

        progress_bar.set_postfix({'loss': loss.item()})
        
    return total_loss / len(train_loader)

def validate_clip(model, val_loader, device):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(val_loader, desc='Validating')
    
    with torch.no_grad():  
        for batch_idx, (images, texts, labels) in enumerate(progress_bar):
            images = images.to(device)
            text_tokens = clip.tokenize(texts).to(device)
            
            # 前向传播
            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)
            
            logits_per_image = image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            loss = (torch.nn.functional.cross_entropy(logits_per_image, ground_truth) + 
                    torch.nn.functional.cross_entropy(logits_per_text, ground_truth)) / 2
            
            total_loss += loss.item()
            progress_bar.set_postfix({'val_loss': loss.item()})
            
            # 记录验证指标
            wandb.log({
                "val_batch_loss": loss.item(),
                "val_image_text_similarity": logits_per_image.mean().item(),
            })
    
    return total_loss / len(val_loader)

def main():
    global json_dir, model_name, fine_tune_method, learning_rate, batch_size, num_epochs, lora_rank, lora_alpha, lora_dropout, save_dir, coarse_weight, fine_weight, label_dict_path, aug_dir, aug_types
    
    # 新增：解析命令行参数
    parser = argparse.ArgumentParser(description="Train CLIP with flexible parameters.")
    parser.add_argument("--json_dir", type=str, default=json_dir, help="Path to JSON dataset directory")
    parser.add_argument("--model_name", type=str, default=model_name, choices=["RN50", "ViT-B/32", "ViT-B/16", "ViT-L/14"], help="CLIP model name")
    parser.add_argument("--fine_tune_method", type=str, default=fine_tune_method, choices=["full", "lora"], help="Fine-tuning method")
    parser.add_argument("--learning_rate", type=float, default=learning_rate, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=batch_size, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=num_epochs, help="Number of epochs")
    parser.add_argument("--lora_rank", type=int, default=lora_rank, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=lora_alpha, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=lora_dropout, help="LoRA dropout rate")
    parser.add_argument("--save_dir", type=str, default=save_dir, help="Model save directory")
    parser.add_argument("--coarse_weight", type=float, default=coarse_weight, help="Coarse-grained loss weight")
    parser.add_argument("--label_dict_path", type=str, default=label_dict_path, help="Path to label dictionary")
    parser.add_argument("--aug_dir", type=str, default=aug_dir, help="Path to augmented images directory")
    args = parser.parse_args()

    # 覆盖 config 参数
    json_dir = args.json_dir
    model_name = args.model_name
    fine_tune_method = args.fine_tune_method
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    save_dir = args.save_dir
    coarse_weight = args.coarse_weight
    label_dict_path = args.label_dict_path
    aug_dir = args.aug_dir
    
    print(f"Training CLIP model with parameters:", args)

    # 初始化 WandB（使用动态参数）
    wandb.init(
        project=wandb_project,
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "model": model_name,
            "fine_tune_method": fine_tune_method,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        }
    )
    
    
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载模型
    model, preprocess = clip.load(model_name, device=device)
    
    # 根据配置选择微调方式
    if fine_tune_method == "lora":
        # 配置 LoRA
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules[model_name],
            lora_dropout=lora_dropout,
            bias="none",
        )
        # 应用 LoRA 到模型
        model = get_peft_model(model, lora_config).float().to(device)
        print(f"Using LoRA for fine-tuning on model: {model_name}")
    else:
        print("Using full fine-tuning.")
    
    # 准备数据集
    train_dataset = TerrainDataset(json_dir, transform=preprocess)

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers
    # )
    
    print(f"Training set size: {len(train_dataset)}")
    # print(f"Validation set size: {len(val_dataset)}")
    
    # 设置优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=optimizer_betas,
        eps=optimizer_eps,
        weight_decay=optimizer_weight_decay
    )

    # 初始化WandB
    wandb.init(
        project=wandb_project,
        config=wandb_config
    )
    
    # 训练循环
    for epoch in range(num_epochs):
        
        # 每10个epoch调整coarse_weight
        # if (epoch + 1) % 50 == 0 and coarse_weight > 0.1:  # 确保coarse_weight不会小于0.1
        #     coarse_weight -= 0.1  # 每次减少0.1
        #     fine_weight = 1 - coarse_weight  # 更新fine_weight
        
        train_loss = train_clip(model, train_loader, optimizer, device, epoch)
        # val_loss = validate_clip(model, val_loader, device)
        
        # 记录每个epoch的指标
        wandb.log({
            "epoch": epoch,
            "average_loss": train_loss,
            # "val_loss": val_loss,
        })

        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}")
        
        # 每5个epoch保存一次模型
        if (epoch + 1) % save_interval == 0:
            save_path = f'{save_dir}/clip_terrain_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                # 'val_loss': val_loss,
                'lora_config': lora_config if fine_tune_method == "lora" else None,  # 保存LoRA配置
            }, save_path)

            print(f"Checkpoint saved to {save_path}")
    
    wandb.finish()

if __name__ == "__main__":
    main()