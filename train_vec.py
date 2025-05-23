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
fonts_dir = "./font_ds/fonts"            # 字体文件夹路径
text_file = "./font_ds/cleaned_text.txt" # 文本文件路径
chars_file = "./font_ds/chars.txt"                  # 常用字文件路径

random.seed(42)

# 初始化 FontSampler，同时会将字体分为 train/test 两类
sampler = FontSampler(fonts_dir, text_file, chars_file, max_fonts=40, font_size=76, train_ratio=0.8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# 定义模型并迁移到设备上，输出嵌入向量
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(device)
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
# def compute_loss(style_vecs, group_size):
#     """
#     计算交叉熵损失。

#     :param style_vecs: 向量序列，由模型生成
#     :param group_size: 每组的大小
#     :return: 每行的交叉熵平均值，作为最终的 Loss
#     """
#     # 计算相似度矩阵
#     similarity_matrix = torch.matmul(style_vecs, style_vecs.T)

#     # 创建目标矩阵
#     target_matrix = torch.zeros_like(similarity_matrix)
#     for i in range(0, len(style_vecs), group_size):
#         target_matrix[i:i+group_size, i:i+group_size] = 1.0 / group_size

#     # 对相似度矩阵的每一行应用 log_softmax
#     log_softmax_matrix = F.log_softmax(similarity_matrix, dim=1)

#     # 计算每一行的 KL 散度
#     losses = []
#     for i in range(len(style_vecs)):
#         row_loss = F.kl_div(log_softmax_matrix[i], target_matrix[i], reduction='sum')
#         losses.append(row_loss)

#     # 计算平均损失
#     loss = torch.stack(losses).mean()

#     # 以 1e-2 的概率输出相似度矩阵（对每行 softmax 后的结果）
#     if random.random() < 1e-2:
#         softmax_matrix = F.softmax(similarity_matrix, dim=1)
#         print("Similarity Matrix (softmax applied):")
#         print(softmax_matrix)

#     return loss

def compute_loss_and_acc(style_vecs, group_size):
    """
    计算交叉熵损失和准确率。

    :param style_vecs: 向量序列，由模型生成
    :param group_size: 每组的大小
    :return: loss 和 acc
    """
    # 计算相似度矩阵
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

    # 计算准确率
    correct = 0
    total = 0
    for i in range(0, len(style_vecs), group_size):
        group_probs = similarity_matrix[i:i+group_size, :]  # 当前组的概率分布

        # 对每一行分别计算
        for row_idx in range(group_size):
            row_probs = group_probs[row_idx]  # 当前行的概率分布
            topk_indices = row_probs.topk(group_size - 1).indices  # 选择概率最高的 group_size-1 个选项

            # 判断 topk_indices 是否属于同一组
            correct += (topk_indices // group_size == i // group_size).sum().item()
            total += group_size - 1  # 每行有 group_size-1 个目标

    acc = correct / total

    return loss, acc

# --------------------------
# 训练和验证步骤（loss 基于 word2vec 风格的 compute_loss）
def train_step(model, epoch, data_loader, optimizer, group_size):
    model.train()
    total_loss = 0
    total_acc = 0
    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch + 1} - Training', leave=True)
    for batch in progress_bar:
        # Flatten the batch into a single tensor
        flattened_batch = [img.to(device) for sample in batch for img in sample]  # Flatten the nested list
        batch_tensor = torch.stack(flattened_batch).squeeze(1)  # Shape: [total_images_in_batch, C, H, W]

        # Pass the entire batch through the model
        style_vecs = model(batch_tensor)  # Shape: [total_images_in_batch, embedding_dim]

        # Compute the loss and accuracy
        loss, acc = compute_loss_and_acc(style_vecs, group_size)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc
        progress_bar.set_postfix(loss=loss.item(), acc=acc)
    progress_bar.close()

    return total_loss / len(data_loader), total_acc / len(data_loader)

def validate(model, data_loader, group_size):
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Validating", leave=True)
        for batch in progress_bar:
            # Flatten the batch into a single tensor
            flattened_batch = [img.to(device) for sample in batch for img in sample]  # Flatten the nested list
            batch_tensor = torch.stack(flattened_batch).squeeze(1)  # Shape: [total_images_in_batch, C, H, W]

            # Pass the entire batch through the model
            style_vecs = model(batch_tensor)  # Shape: [total_images_in_batch, embedding_dim]

            # Compute the loss and accuracy
            loss, acc = compute_loss_and_acc(style_vecs, group_size)

            total_loss += loss.item()
            total_acc += acc
            progress_bar.set_postfix(loss=loss.item(), acc=acc)
        progress_bar.close()

    return total_loss / len(data_loader), total_acc / len(data_loader)

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
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08)

# 迭代次数，可根据需求调整
num_epochs = 128
epoch_length = 64  # 每个 epoch 中的 batch 个数

# 假设每次采样返回的样本中，同一字体的样本数等于 sample_cnt，此处作为 group_size
font_cnt = 4
sample_cnt = 6
batch_size = 16  # 每个批次的样本数

def sample(sampler, font_cnt, sample_cnt, sample_source):
    sample = sampler.sample(font_cnt=font_cnt, sample_cnt=sample_cnt, sample_source=sample_source)
    return sample

import concurrent.futures

for epoch in range(num_epochs):
    # 收集一个 epoch 所需的所有训练样本
    train_samples = []
    val_samples = []

    # 使用多线程采样所有数据
    total_samples = epoch_length * batch_size
    val_samples_count = total_samples // 16  # 1/16 的数据用于验证
    train_samples_count = total_samples

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        val_futures = [executor.submit(sample, sampler, font_cnt, sample_cnt, "test") for _ in range(val_samples_count)]
        val_samples = [future.result() for future in tqdm(val_futures, desc=f"Epoch {epoch + 1} - Collecting val samples")]

        train_futures = [executor.submit(sample, sampler, font_cnt, sample_cnt, "train") for _ in range(train_samples_count)]
        train_samples = [future.result() for future in tqdm(train_futures, desc=f"Epoch {epoch + 1} - Collecting train samples")]

    # 将采样结果重新排布为 [epoch_length, batch_size] 的格式
    train_batches = []
    for i in range(epoch_length):
        batch_samples = train_samples[i * batch_size:(i + 1) * batch_size]
        train_batches.append(batch_samples)

    val_batches = []
    val_length = len(val_samples) // batch_size
    for i in range(val_length):
        batch_samples = val_samples[i * batch_size:(i + 1) * batch_size]
        val_batches.append(batch_samples)

    # 创建训练集 Dataset 和 DataLoader
    train_dataset = FontDataset(train_batches, transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # 创建验证集 Dataset 和 DataLoader
    val_dataset = FontDataset(val_batches, transform=data_transforms)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    train_loss, train_acc = train_step(model, epoch, train_loader, optimizer, sample_cnt)
    val_loss, val_acc = validate(model, val_loader, sample_cnt)

    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 保存模型
    model_save_path = f'font_identifier_model_epoch_{epoch + 1}.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")