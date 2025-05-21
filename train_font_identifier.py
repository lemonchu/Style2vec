import os
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm
from consts import TRAIN_TEST_IMAGES_DIR
from datasets import load_dataset
from torch.utils.data import Dataset

# Transformations for the image data
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), # Convert images to grayscale with 3 channels
    transforms.RandomCrop((224, 224)), # Resize images to the expected input size of the model
    transforms.ToTensor(), # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize with ImageNet stats
])

# Load dataset and split data
ds_full = load_dataset("gaborcselle/font-examples")
split_ds = ds_full['train'].train_test_split(test_size=0.2, seed=42)

# Custom wrapper to adapt Hugging Face dataset structure (fields: image, label)
class HFDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']  # Expected to be of PIL.Image type
        label = sample['label']
        if self.transform:
            image = self.transform(image)
        return image, label

# Create datasets
image_datasets = {
    'train': HFDataset(split_ds['train'], data_transforms),
    'test': HFDataset(split_ds['test'], data_transforms)
}

# Create dataloaders
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=16, shuffle=True),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=16, shuffle=True)
}

# 设置使用CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model and move it to device
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)

# Modify the last fully connected layer to match the number of font classes you have
num_classes = len(image_datasets['train'])
model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)

# Define the loss function
criterion = torch.nn.CrossEntropyLoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9,0.999), eps=1e-08)

# Function to perform a training step with progress bar
def train_step(model, data_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc='Training', leave=True)
    for inputs, targets in progress_bar:
        # 将数据转移到对应设备上
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
            # 将数据转移到对应设备上
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

print(image_datasets['train'])

# Training loop with progress bar for epochs
num_epochs = 50  # Replace with the number of epochs you'd like to train for
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss = train_step(model, dataloaders["train"], criterion, optimizer)
    val_loss, val_accuracy = validate(model, dataloaders["test"], criterion)
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Save the model to disk
torch.save(model.state_dict(), 'font_identifier_model.pth')


