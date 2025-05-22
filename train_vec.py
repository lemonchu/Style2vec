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
fonts_dir = "./font_ds/fonts"            # 字体文件夹路径
text_file = "./font_ds/cleaned_test.txt" # 文本文件路径
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
    计算交叉熵损失。

    :param style_vecs: 向量序列，由模型生成
    :param group_size: 每组的大小
    :return: 每行的交叉熵平均值，作为最终的 Loss
    """
    # 计算相似度矩阵
    similarity_matrix = torch.matmul(style_vecs, style_vecs.T)

    # 对相似度矩阵的第 i 行做 soft max
    softmax_matrix = F.softmax(similarity_matrix, dim=1)

    # 创建目标矩阵
    target_matrix = torch.zeros_like(softmax_matrix)
    for i in range(0, len(style_vecs), group_size):
        target_matrix[i:i+group_size, i:i+group_size] = 1/group_size

    # 计算交叉熵损失
    loss = F.cross_entropy(softmax_matrix, target_matrix)

    return loss

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


import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# 假设 data_transforms 是一个预定义的转换函数
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 定义一个简单的 Dataset 类来处理样本
class FontDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, font_id = self.samples[idx]
        if self.transform:
            img = self.transform(img)
        return img, font_id


# 定义优化器（只包含 model 参数）
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-08)

# 迭代次数，可根据需求调整
num_iters = 1000
random.seed(42)

# 假设每次采样返回的样本中，同一字体的样本数等于 sample_cnt，此处作为 group_size
font_cnt = 8
sample_cnt = 8
batch_size = 16  # 每个批次的样本数

for iter in range(num_iters):
    # 收集 batch_size 个训练样本
    train_samples = []
    for _ in range(batch_size):
        sample = sampler.sample(font_cnt=font_cnt, sample_cnt=sample_cnt, sample_source="train")
        train_samples.extend(sample)

    # 收集 batch_size 个测试样本
    test_samples = []
    for _ in range(batch_size):
        sample = sampler.sample(font_cnt=font_cnt, sample_cnt=sample_cnt, sample_source="test")
        test_samples.extend(sample)

    # 创建 Dataset 和 DataLoader
    train_dataset = FontDataset(train_samples, transform=data_transforms)
    test_dataset = FontDataset(test_samples, transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 执行本次迭代的训练与验证
    train_loss = train_step(model, train_loader, optimizer)
    val_loss = validate(model, test_loader)
    print(f"Iter {iter + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), 'font_identifier_model.pth')