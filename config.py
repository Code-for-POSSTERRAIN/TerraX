# Config.py

# 训练标注文件路径
# json_dir = "/home4/xuchuan/gpt_annotation/2nd_annotation_JACKAL_results"
json_dir = "/home4/xuchuan/gpt_annotation/cleaned_annotation_RSCD_results"

# 测试图片保存路径
data_dir = "/home4/xinhai/new_dataset/RSCD/test"

# 增强图片保存路径
aug_dir = "/home4/xuchuan/imgaug_results/our_data_imgaug_results"
aug_types = ["brightness", "contrast", "dark_severe", "dark_slight", "dark_smooth", "fog", "motion blur", "rain", "saturate", "snow", "snowflakes"]

# 模型与结果保存路径
save_dir = "ckp/ViT_lora_only_text_our_data_new_dual_0.6_aug"
save_interval = 5  # 每5个epoch保存一次模型
save_test_file = "classification_results.json"

# 细粒度属性标签字典路径
label_dict_path = "/home4/TerraX/label_dict.json"

# 训练参数
batch_size = 64
num_epochs = 100
learning_rate = 5e-5

# Loss参数
coarse_weight = 0.5  # 粗粒度标签的权重
fine_weight = 1 - coarse_weight  # 细粒度标签的权重
temperature = 1.0  # 温度参数

# 模型参数
# RN50, RN101, RN50x4, RN50x16, RN50x64, ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px
model_name = "ViT-B/32"

# 微调方式：直接微调（full）或 LoRA 微调（lora）
fine_tune_method = "lora"  # 可选 "full" 或 "lora"

# 优化器参数
optimizer_betas = (0.9, 0.98)
optimizer_eps = 1e-6
optimizer_weight_decay = 0.2

# 数据加载参数
num_workers = 4

# LoRA 微调参数（仅在 fine_tune_method = "lora" 时生效）
lora_rank = 8  # LoRA 的秩
lora_alpha = 16  # LoRA 的 alpha 参数
lora_dropout = 0.1  # LoRA 的 dropout 率

# 不同模型的 LoRA 参数层配置
lora_target_modules = {
    "RN50": [
        # 视觉部分的卷积层
        "visual.conv1",
        "visual.layer1.0.conv1",
        "visual.layer1.0.conv2",
        "visual.layer1.0.conv3",
        "visual.layer1.0.downsample.0",
        "visual.layer2.0.conv1",
        "visual.layer2.0.conv2",
        "visual.layer2.0.conv3",
        "visual.layer2.0.downsample.0",
        "visual.layer3.0.conv1",
        "visual.layer3.0.conv2",
        "visual.layer3.0.conv3",
        "visual.layer3.0.downsample.0",
        "visual.layer4.0.conv1",
        "visual.layer4.0.conv2",
        "visual.layer4.0.conv3",
        "visual.layer4.0.downsample.0",
        
        # 视觉部分的注意力池化层
        "visual.attnpool.k_proj",
        "visual.attnpool.q_proj",
        "visual.attnpool.v_proj",
        "visual.attnpool.c_proj",
        
        # 文本部分的注意力机制
        "transformer.resblocks.0.attn.in_proj_weight",
        "transformer.resblocks.0.attn.out_proj",
        "transformer.resblocks.1.attn.in_proj_weight",
        "transformer.resblocks.1.attn.out_proj",
        "transformer.resblocks.2.attn.in_proj_weight",
        "transformer.resblocks.2.attn.out_proj",
        "transformer.resblocks.3.attn.in_proj_weight",
        "transformer.resblocks.3.attn.out_proj",
        "transformer.resblocks.4.attn.in_proj_weight",
        "transformer.resblocks.4.attn.out_proj",
        "transformer.resblocks.5.attn.in_proj_weight",
        "transformer.resblocks.5.attn.out_proj",
        "transformer.resblocks.6.attn.in_proj_weight",
        "transformer.resblocks.6.attn.out_proj",
        "transformer.resblocks.7.attn.in_proj_weight",
        "transformer.resblocks.7.attn.out_proj",
        "transformer.resblocks.8.attn.in_proj_weight",
        "transformer.resblocks.8.attn.out_proj",
        "transformer.resblocks.9.attn.in_proj_weight",
        "transformer.resblocks.9.attn.out_proj",
        "transformer.resblocks.10.attn.in_proj_weight",
        "transformer.resblocks.10.attn.out_proj",
        "transformer.resblocks.11.attn.in_proj_weight",
        "transformer.resblocks.11.attn.out_proj",
        
        # 文本部分的 MLP 层
        "transformer.resblocks.0.mlp.c_fc",
        "transformer.resblocks.0.mlp.c_proj",
        "transformer.resblocks.1.mlp.c_fc",
        "transformer.resblocks.1.mlp.c_proj",
        "transformer.resblocks.2.mlp.c_fc",
        "transformer.resblocks.2.mlp.c_proj",
        "transformer.resblocks.3.mlp.c_fc",
        "transformer.resblocks.3.mlp.c_proj",
        "transformer.resblocks.4.mlp.c_fc",
        "transformer.resblocks.4.mlp.c_proj",
        "transformer.resblocks.5.mlp.c_fc",
        "transformer.resblocks.5.mlp.c_proj",
        "transformer.resblocks.6.mlp.c_fc",
        "transformer.resblocks.6.mlp.c_proj",
        "transformer.resblocks.7.mlp.c_fc",
        "transformer.resblocks.7.mlp.c_proj",
        "transformer.resblocks.8.mlp.c_fc",
        "transformer.resblocks.8.mlp.c_proj",
        "transformer.resblocks.9.mlp.c_fc",
        "transformer.resblocks.9.mlp.c_proj",
        "transformer.resblocks.10.mlp.c_fc",
        "transformer.resblocks.10.mlp.c_proj",
        "transformer.resblocks.11.mlp.c_fc",
        "transformer.resblocks.11.mlp.c_proj",
    ],  # ResNet-50
    
    "ViT-B/32": [
        # 视觉部分的注意力机制
        "visual.transformer.resblocks.0.attn.in_proj_weight",  # QKV 投影
        "visual.transformer.resblocks.0.attn.out_proj",       # 输出投影
        "visual.transformer.resblocks.1.attn.in_proj_weight",
        "visual.transformer.resblocks.1.attn.out_proj",
        "visual.transformer.resblocks.2.attn.in_proj_weight",
        "visual.transformer.resblocks.2.attn.out_proj",
        "visual.transformer.resblocks.3.attn.in_proj_weight",
        "visual.transformer.resblocks.3.attn.out_proj",
        "visual.transformer.resblocks.4.attn.in_proj_weight",
        "visual.transformer.resblocks.4.attn.out_proj",
        "visual.transformer.resblocks.5.attn.in_proj_weight",
        "visual.transformer.resblocks.5.attn.out_proj",
        "visual.transformer.resblocks.6.attn.in_proj_weight",
        "visual.transformer.resblocks.6.attn.out_proj",
        "visual.transformer.resblocks.7.attn.in_proj_weight",
        "visual.transformer.resblocks.7.attn.out_proj",
        "visual.transformer.resblocks.8.attn.in_proj_weight",
        "visual.transformer.resblocks.8.attn.out_proj",
        "visual.transformer.resblocks.9.attn.in_proj_weight",
        "visual.transformer.resblocks.9.attn.out_proj",
        "visual.transformer.resblocks.10.attn.in_proj_weight",
        "visual.transformer.resblocks.10.attn.out_proj",
        "visual.transformer.resblocks.11.attn.in_proj_weight",
        "visual.transformer.resblocks.11.attn.out_proj",
        
        # 视觉部分的前馈神经网络（FFN）
        "visual.transformer.resblocks.0.mlp.fc1",
        "visual.transformer.resblocks.0.mlp.fc2",
        "visual.transformer.resblocks.1.mlp.fc1",
        "visual.transformer.resblocks.1.mlp.fc2",
        "visual.transformer.resblocks.2.mlp.fc1",
        "visual.transformer.resblocks.2.mlp.fc2",
        "visual.transformer.resblocks.3.mlp.fc1",
        "visual.transformer.resblocks.3.mlp.fc2",
        "visual.transformer.resblocks.4.mlp.fc1",
        "visual.transformer.resblocks.4.mlp.fc2",
        "visual.transformer.resblocks.5.mlp.fc1",
        "visual.transformer.resblocks.5.mlp.fc2",
        "visual.transformer.resblocks.6.mlp.fc1",
        "visual.transformer.resblocks.6.mlp.fc2",
        "visual.transformer.resblocks.7.mlp.fc1",
        "visual.transformer.resblocks.7.mlp.fc2",
        "visual.transformer.resblocks.8.mlp.fc1",
        "visual.transformer.resblocks.8.mlp.fc2",
        "visual.transformer.resblocks.9.mlp.fc1",
        "visual.transformer.resblocks.9.mlp.fc2",
        "visual.transformer.resblocks.10.mlp.fc1",
        "visual.transformer.resblocks.10.mlp.fc2",
        "visual.transformer.resblocks.11.mlp.fc1",
        "visual.transformer.resblocks.11.mlp.fc2",
        
        # 注意力机制中的 QKV 投影层
        "transformer.resblocks.0.attn.in_proj_weight",
        "transformer.resblocks.0.attn.out_proj",
        "transformer.resblocks.1.attn.in_proj_weight",
        "transformer.resblocks.1.attn.out_proj",
        "transformer.resblocks.2.attn.in_proj_weight",
        "transformer.resblocks.2.attn.out_proj",
        "transformer.resblocks.3.attn.in_proj_weight",
        "transformer.resblocks.3.attn.out_proj",
        "transformer.resblocks.4.attn.in_proj_weight",
        "transformer.resblocks.4.attn.out_proj",
        "transformer.resblocks.5.attn.in_proj_weight",
        "transformer.resblocks.5.attn.out_proj",
        "transformer.resblocks.6.attn.in_proj_weight",
        "transformer.resblocks.6.attn.out_proj",
        "transformer.resblocks.7.attn.in_proj_weight",
        "transformer.resblocks.7.attn.out_proj",
        "transformer.resblocks.8.attn.in_proj_weight",
        "transformer.resblocks.8.attn.out_proj",
        "transformer.resblocks.9.attn.in_proj_weight",
        "transformer.resblocks.9.attn.out_proj",
        "transformer.resblocks.10.attn.in_proj_weight",
        "transformer.resblocks.10.attn.out_proj",
        "transformer.resblocks.11.attn.in_proj_weight",
        "transformer.resblocks.11.attn.out_proj",
        
        # 前馈神经网络（FFN）中的全连接层
        "transformer.resblocks.0.mlp.c_fc",
        "transformer.resblocks.0.mlp.c_proj",
        "transformer.resblocks.1.mlp.c_fc",
        "transformer.resblocks.1.mlp.c_proj",
        "transformer.resblocks.2.mlp.c_fc",
        "transformer.resblocks.2.mlp.c_proj",
        "transformer.resblocks.3.mlp.c_fc",
        "transformer.resblocks.3.mlp.c_proj",
        "transformer.resblocks.4.mlp.c_fc",
        "transformer.resblocks.4.mlp.c_proj",
        "transformer.resblocks.5.mlp.c_fc",
        "transformer.resblocks.5.mlp.c_proj",
        "transformer.resblocks.6.mlp.c_fc",
        "transformer.resblocks.6.mlp.c_proj",
        "transformer.resblocks.7.mlp.c_fc",
        "transformer.resblocks.7.mlp.c_proj",
        "transformer.resblocks.8.mlp.c_fc",
        "transformer.resblocks.8.mlp.c_proj",
        "transformer.resblocks.9.mlp.c_fc",
        "transformer.resblocks.9.mlp.c_proj",
        "transformer.resblocks.10.mlp.c_fc",
        "transformer.resblocks.10.mlp.c_proj",
        "transformer.resblocks.11.mlp.c_fc",
        "transformer.resblocks.11.mlp.c_proj",
    ],  # ViT
}

# WandB 配置
wandb_project = "CLIP-LoRA"
wandb_config = {
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "model": model_name,
    "fine_tune_method": fine_tune_method,
    "lora_rank": lora_rank,
    "lora_alpha": lora_alpha,
    "lora_dropout": lora_dropout,
}

cross_sets = "JA_RS"
cross_labels = {
    "JA_RS": ["asphalt", "gravel"],
    "RS_JA": ["asphalt", "gravel"],
    "JA_PO": ["asphalt", "brick", "grass", "gravel", "sand"],
    "PO_JA": ["asphalt", "brick", "grass", "gravel", "sand"],
    "PO_RS": ["asphalt", "concrete", "gravel"],
    "RS_PO": ["asphalt", "concrete", "gravel"],
}