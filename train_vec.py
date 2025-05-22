import os
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import FontSampler

# 字体相关路径
fonts_dir = "../font_ds/fonts"       # 字体文件夹路径
text_file = "../font_ds/cleaned_test.txt"      # 文本文件路径
chars_file = "chars.txt"    # 常用字文件路径

# 初始化 FontSampler，同时会将字体分为 train/test 两类
sampler = FontSampler(fonts_dir, text_file, chars_file, max_fonts=None, font_size=76, train_ratio=0.8)

# Transformations for the image data
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 转为 3 通道灰度图像
    transforms.ToTensor(),                          # 转为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 标准化
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model and move it to device
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)

# 设置嵌入向量相关的超参数
embedding_dim = 128  # 输出嵌入向量的维度
hidden_dim = 256     # 隐藏层维度

# 修改最后全连接层，使用一个 Sequential 网络输出嵌入向量
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, hidden_dim),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(hidden_dim, embedding_dim)
).to(device)

# 自定义 Criterion
class CustomCriterion(nn.Module):
    def __init__(self):
        super(CustomCriterion, self).__init__()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets):
        # 可在此加入其他损失项，现仅使用交叉熵
        return self.ce(outputs, targets)

criterion = CustomCriterion()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-08)

# Function to perform a training step with progress bar
def train_step(model, data_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc='Training', leave=True)
    for inputs, targets in progress_bar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    progress_bar.close()
    return total_loss / len(data_loader)

# Function to perform a validation step with progress bar
def validate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    progress_bar = tqdm(data_loader, desc='Validation', leave=False)
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            progress_bar.set_postfix(loss=loss.item())
    progress_bar.close()
    return total_loss / len(data_loader), correct / len(data_loader.dataset)

# 预留模板函数：处理采样数据（例如数据增强、拼接等自定义逻辑）
def process_sample(samples):
    # TODO: 在此处实现你自己的 sample 数据处理逻辑
    return samples

num_iters = 1000  # 迭代次数，根据需求调整
random.seed(42)

for iter in range(num_iters):
    # 通过 FontSampler 获取训练和测试样本
    # sampler.sample 返回的是 [(image, font_id), ...]
    train_samples = sampler.sample(sample_cnt=8, sample_source="train")
    test_samples = sampler.sample(sample_cnt=8, sample_source="test")
    
    # 调用自定义的数据处理逻辑（预留模板）
    train_samples = process_sample(train_samples)
    test_samples = process_sample(test_samples)
    
    # 对每个样本的图像进行 transformation 处理
    train_samples = [(data_transforms(img), font_id) for img, font_id in train_samples]
    test_samples = [(data_transforms(img), font_id) for img, font_id in test_samples]
    
    # 直接利用 Python 列表作为数据集传入 DataLoader
    train_loader = DataLoader(train_samples, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_samples, batch_size=16, shuffle=True)
    
    # 执行本次迭代的训练与验证
    train_loss = train_step(model, train_loader, criterion, optimizer)
    val_loss, val_accuracy = validate(model, test_loader, criterion)
    print(f"Iter {iter+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# 保存模型
torch.save(model.state_dict(), 'font_identifier_model.pth')


