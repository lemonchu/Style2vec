import os
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from utils import FontSampler

# --------------------------
# 配置字体相关路径
fonts_dir = "./font_ds/fonts"
text_file = "./font_ds/cleaned_text.txt"
chars_file = "./font_ds/chars.txt"

# 初始化 FontSampler，同时会将字体分为 train/test 两类
sampler = FontSampler(fonts_dir, text_file, chars_file, max_fonts=16, font_size=76, train_ratio=0.5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
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
    # 计算相似度矩阵 (Emb 点积)
    similarity_matrix = torch.matmul(style_vecs, style_vecs.T)

    # 创建目标矩阵
    target_matrix = torch.zeros_like(similarity_matrix)
    for i in range(0, len(style_vecs), group_size):
        target_matrix[i:i+group_size, i:i+group_size] = 1.0 / group_size

    # 对相似度矩阵的每一行应用 log_softmax
    log_softmax_matrix = F.log_softmax(similarity_matrix, dim=1)

    # 计算每一行的 KL 散度
    losses = []
    for i in range(len(style_vecs)):
        row_loss = F.kl_div(log_softmax_matrix[i], target_matrix[i], reduction='sum')
        losses.append(row_loss)

    # 计算平均损失
    loss = torch.stack(losses).mean()

    # 以 1e-2 的概率输出相似度矩阵（对每行 softmax 后的结果）
    if random.random() < 1e-2:
        softmax_matrix = F.softmax(similarity_matrix, dim=1)
        print("Similarity Matrix (softmax applied):")
        print(softmax_matrix)

    return loss

# --------------------------
# 训练和验证步骤（loss 基于 word2vec 风格的 compute_loss）
def train_step(model, data_loader, optimizer, group_size):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc='Training', leave=True)
    for batch in progress_bar:
        loss = 0
        print('    len(batch)', len(batch))
        for sample in batch:
            sample = [img.to(device) for img in sample]
            style_vecs = torch.stack([model(img) for img in sample])
            print('    style_vecs.shape', style_vecs.shape)
            loss += compute_loss(style_vecs, group_size)
        
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

# Transformations for the image 数据
data_transforms = transforms.Compose([
    transforms.ToTensor(),  # 转为张量
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 将单通道图像复制为3通道
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 标准化
])

# 定义一个简单的 Dataset 类来处理样本
class FontDataset(Dataset):
    def __init__(self, batchs, transform=None):
        self.batchs = batchs
        self.transform = transform

    def __len__(self):
        return len(self.batchs)

    def __getitem__(self, idx):
        batch = self.batchs[idx]
        if self.transform:
            batch = [[self.transform(img) for img in inner_list] for inner_list in batch]
        return batch

# 定义优化器（只包含 model 参数）
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-08)

# 迭代次数，可根据需求调整
num_epochs = 1024
epoch_length = 17  # 每个 epoch 中的 batch 个数
random.seed(42)

# 假设每次采样返回的样本中，同一字体的样本数等于 sample_cnt，此处作为 group_size
font_cnt = 2
sample_cnt = 2
batch_size = 13  # 每个批次的样本数

def sample(sampler, font_cnt, sample_cnt, sample_source):
    sample = sampler.sample(font_cnt=font_cnt, sample_cnt=sample_cnt, sample_source=sample_source)
    return sample

import concurrent.futures

for epoch in range(num_epochs):
    # 收集一个 epoch 所需的所有训练样本
    train_samples = []
    test_samples = []

    for _ in range(epoch_length):
        batch_samples = []
        for _ in range(batch_size):
            batch_samples.append(sample(sampler, font_cnt, sample_cnt, "train"))
        train_samples.append(batch_samples)

    # with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    #     futures = [executor.submit(sample, sampler, font_cnt, sample_cnt, "train") for _ in range(epoch_length * batch_size)]
    #     for future in tqdm(futures, desc=f"Epoch {epoch + 1} - Collecting train samples"):
    #         train_samples.append(future.result())
    
    # with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    #     futures = [executor.submit(sample, sampler, font_cnt, sample_cnt, "test") for _ in range(epoch_length * batch_size)]
    #     for future in tqdm(futures, desc=f"Epoch {epoch + 1} - Collecting test samples"):
    #         test_samples.append(future.result())


    # 创建 Dataset 和 DataLoader
    train_dataset = FontDataset(train_samples, transform=data_transforms)
    # test_dataset = FontDataset(test_samples, transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 执行本次 epoch 的训练与验证
    train_loss = train_step(model, train_loader, optimizer, sample_cnt)
    # val_loss = validate(model, test_loader, sample_cnt)
    print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), 'font_identifier_model.pth')