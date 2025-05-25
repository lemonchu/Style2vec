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

sampler = FontSampler(fonts_dir, text_file, chars_file, font_size=76)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# 定义模型
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(device)
embedding_dim = 128  # 输出嵌入向量的维度
hidden_dim = 256     # 隐藏层维度

# 修改最后全连接层，直接输出嵌入向量
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, hidden_dim),
    nn.ReLU(inplace=True),
    nn.Linear(hidden_dim, embedding_dim)
).to(device)

# 定义优化器（只包含 model 参数）
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-08)


def compute_loss_and_acc(style_vecs, group_size, alpha = 4.0):
    """
    计算交叉熵损失和准确率。
    对于每一行，重新生成一个 tensor，直接去除对角线（即自身的分数）。

    :param style_vecs: 向量序列，由模型生成，形状为 [N, embedding_dim]
    :param group_size: 每组的大小
    :return: loss 和 acc
    """

    # 对 style_vecs 进行 L2 标准化
    style_vecs = F.normalize(style_vecs, p=2, dim=1)
    
    # 计算点积，然后对 0 取 max，再提升到 alpha 次方
    dot_prod = torch.matmul(style_vecs, style_vecs.T)
    similarity_matrix = torch.clamp(dot_prod, min=1e-8) ** alpha

    N = similarity_matrix.size(0)
    losses = []
    correct = 0

    for i in range(N):
        row = similarity_matrix[i]  # shape: [N]
        # 重新构造一个 tensor，去除自身的分数（第 i 个元素）
        new_row = torch.cat((row[:i], row[i+1:]))  # shape: [N-1]
        # 对 new_row 进行归一化
        new_row = F.normalize(new_row, p=1, dim=0)
        
        # 构造目标分布：对于当前行所属的组（group_start 到 group_end-1），除去自身，每个目标均为 1/(group_size-1)
        target = torch.zeros_like(new_row)
        group_start = (i // group_size) * group_size
        group_end = group_start + group_size -1
        target[group_start:group_end] = 1.0 / (group_size - 1)
        
        # 计算 KL 散度损失
        row_loss = F.kl_div(new_row.log(), target, reduction='sum')
        losses.append(row_loss)

        # 计算准确率：
        # 从 new_row 选取 top-(group_size-1)，如果这些位置对应的原始索引均落在同一组中，则算作正确
        topk_indices = new_row.topk(group_size - 1).indices
        correct_in_row = ((topk_indices >= group_start) & (topk_indices < group_end)).sum().item()
        correct += correct_in_row

    loss = torch.stack(losses).mean()
    acc = correct / ((group_size - 1) * N)

    return loss, acc

# --------------------------
# 训练和验证步骤 (loss 基于 word2vec 风格的 compute_loss)
def train_step(model, epoch, data_loader, optimizer, batch_size, font_size, group_size):
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

        # Reshape the output to match the expected input shape for compute_loss_and_acc
        style_vecs = style_vecs.view(batch_size, font_size * group_size, -1)  # Shape: [batch_size, group_size, embedding_dim]
        
        # Compute the loss and accuracy
        loss, acc = 0, 0
        for i in range(batch_size):
            sample_loss, sample_acc = compute_loss_and_acc(style_vecs[i], group_size)
            loss += sample_loss
            acc += sample_acc

        loss /= batch_size
        acc /= batch_size

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc
        progress_bar.set_postfix(loss=loss.item(), acc=acc)
    progress_bar.close()

    return total_loss / len(data_loader), total_acc / len(data_loader)

def validate(model, data_loader, batch_size, font_size, group_size):
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

            # Reshape the output to match the expected input shape for compute_loss_and_acc
            style_vecs = style_vecs.view(batch_size, font_size * group_size, -1) # Shape: [batch_size, group_size, embedding_dim]
            
            # Compute the loss and accuracy
            loss, acc = 0, 0
            for i in range(batch_size):
                sample_loss, sample_acc = compute_loss_and_acc(style_vecs[i], group_size)
                loss += sample_loss
                acc += sample_acc

            loss /= batch_size
            acc /= batch_size

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
    
# 迭代次数，可根据需求调整
num_epochs = 32
epoch_length = 64  # 每个 epoch 中的 batch 个数

# 假设每次采样返回的样本中，同一字体的样本数等于 sample_cnt，此处作为 group_size
font_cnt = 4
sample_cnt = 4
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
    val_samples_count = total_samples // 8  # 1/8 的数据用于验证
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

    train_loss, train_acc = train_step(model, epoch, train_loader, optimizer, batch_size, font_cnt, sample_cnt)
    val_loss, val_acc = validate(model, val_loader, batch_size, font_cnt, sample_cnt)

    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 保存模型
    model_save_path = f'font_style2vec_model_epoch_{epoch + 1}.pth'
    torch.save(model, model_save_path)
    print(f"Model saved to {model_save_path}")