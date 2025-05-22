import os
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import FontSampler

# --------------------------
# 配置字体相关路径
fonts_dir = "../font_ds/fonts"            # 字体文件夹路径
text_file = "../font_ds/cleaned_test.txt" # 文本文件路径
chars_file = "chars.txt"                  # 常用字文件路径

# 初始化 FontSampler，同时会将字体分为 train/test 两类
sampler = FontSampler(fonts_dir, text_file, chars_file, max_fonts=None, font_size=76, train_ratio=0.8)

# Transformations for the image 数据
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),    # 转为 3 通道灰度图像
    transforms.ToTensor(),                          # 转为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 标准化
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# 定义模型并迁移到设备上，输出嵌入向量
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
embedding_dim = 128  # 输出嵌入向量的维度
hidden_dim = 256     # 隐藏层维度

# 修改最后全连接层，直接输出嵌入向量
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, hidden_dim),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(hidden_dim, embedding_dim)
).to(device)

# --------------------------
# 定义 compute_loss，基于 word2vec 思路计算 loss
def compute_loss(style_vecs, group_size):
    """
    计算 style2vec 损失：
    1. 计算相似度矩阵：通过 dot product 计算嵌入向量间相似度
    2. 对每一行通过 log_softmax 得到预测分布
    3. 对于每个样本，其正样本来自与其同一组（组内除自身之外的样本）
       损失为负对数概率的平均值
       
    :param style_vecs: 模型输出的嵌入向量, shape [N, embedding_dim]
    :param group_size: 每组样本数 (保证采样返回的列表中，同组样本连续排列)
    :return: scalar loss
    """
    # 计算相似度矩阵，shape: [N, N]
    similarity_matrix = torch.matmul(style_vecs, style_vecs.T)
    # 计算行内预测分布
    log_probs = F.log_softmax(similarity_matrix, dim=1)  # shape: [N, N]
    
    N = style_vecs.shape[0]
    loss_total = 0.0
    count = 0
    num_groups = N // group_size
    for i in range(num_groups):
        group_start = i * group_size
        group_end = group_start + group_size
        indices = list(range(group_start, group_end))
        for j in indices:
            # 取同组内除自身之外的正样本索引
            pos_indices = [k for k in indices if k != j]
            if pos_indices:
                loss_sample = -log_probs[j, pos_indices].mean()
                loss_total += loss_sample
                count += 1
    return loss_total / count if count > 0 else loss_total

# --------------------------
# 预留模板函数：处理采样数据（例如数据增强、拼接等自定义逻辑）
def process_sample(samples):
    # TODO: 在此处实现你自己的 sample 数据处理逻辑
    return samples

# --------------------------
# 训练和验证步骤（loss 基于 word2vec 风格的 compute_loss）
def train_step(model, data_loader, optimizer, group_size):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc='Training', leave=True)
    for inputs, _ in progress_bar:
        inputs = inputs.to(device)
        style_vecs = model(inputs)
        loss = compute_loss(style_vecs, group_size)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    progress_bar.close()
    return total_loss / len(data_loader)

def validate(model, data_loader, group_size):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc='Validation', leave=False)
    with torch.no_grad():
        for inputs, _ in progress_bar:
            inputs = inputs.to(device)
            style_vecs = model(inputs)
            loss = compute_loss(style_vecs, group_size)
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
    progress_bar.close()
    return total_loss / len(data_loader)

# --------------------------
# 定义优化器（只包含 model 参数）
optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-08)

# --------------------------
num_iters = 1000  # 迭代次数，可根据需求调整
random.seed(42)
# 假设每次采样返回的样本中，同一字体的样本数等于 sample_cnt，此处作为 group_size
train_sample_cnt = 8
test_sample_cnt = 2
group_size = 4

for iter in range(num_iters):
    # 通过 FontSampler 获取训练和测试样本
    # sampler.sample 返回 [(image, font_id), ...]，此处忽略 font_id
    train_samples = sampler.sample(sample_cnt=train_sample_cnt, sample_source="train")
    test_samples = sampler.sample(sample_cnt=test_sample_cnt, sample_source="test")
    
    # 调用自定义的数据处理逻辑（预留模板）
    train_samples = process_sample(train_samples)
    test_samples = process_sample(test_samples)
    
    # 对每个样本图像做 transformation 处理
    train_samples = [(data_transforms(img), font_id) for img, font_id in train_samples]
    test_samples = [(data_transforms(img), font_id) for img, font_id in test_samples]
    
    # 利用 Python 列表构造 DataLoader
    train_loader = DataLoader(train_samples, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_samples, batch_size=16, shuffle=True)
    
    # 执行本次迭代的训练与验证
    train_loss = train_step(model, train_loader, optimizer, group_size)
    val_loss = validate(model, test_loader, group_size)
    print(f"Iter {iter+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), 'font_identifier_model.pth')


