import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import time

# Define paths
data_dir = './KneeXray/normalized_train'
label_file = './OA_mask_train/Train.csv'

# Load labels
data = pd.read_csv(label_file)

# Filter labels based on existing images
image_files = set([f.split('_mask')[0] for f in os.listdir(data_dir) if f.endswith('.jpg')])
data['filename'] = data['filename'].str.replace(' ', '')  # Clean spaces
data = data[data['filename'].isin(image_files)]  # Match filenames

# Check if data is empty
if data.empty:
    raise ValueError("No matching filenames found between CSV and image directory.")

# Define Dataset
class OADataset(Dataset):
    def __init__(self, data, img_dir, transform=None):
        self.data = data
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx]['filename'])
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == '__main__':
    # Image transformations with optimized augmentations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Reduce image size
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Prepare data
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = OADataset(train_data, data_dir, transform=transform)
    val_dataset = OADataset(val_data, data_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # Use Pretrained Model - ResNet34
    class PretrainedModel(nn.Module):
        def __init__(self):
            super(PretrainedModel, self).__init__()
            self.model = models.resnet34(pretrained=True)
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 5)
            )

        def forward(self, x):
            return self.model(x)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PretrainedModel().to(device)

    # Loss and optimizer
    class_weights = torch.tensor([1.0, 1.5, 1.2, 1.3, 1.7], device=device)  # Adjust weights for imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Training loop
    num_epochs = 30
    scaler = GradScaler()
    early_stopping_patience = 7
    best_val_loss = float('inf')
    best_model_path = 'best_oa_severity_model.pth'
    gradient_accumulation_steps = 4  # Simulate larger batch size

    train_losses, val_losses = [], []
    val_accuracies = []
    val_tolerance_accuracies = []
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_losses.append(running_loss / len(train_loader))
        train_accuracy = 100 * correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        within_tolerance = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                within_tolerance += ((predicted - labels).abs() <= 1).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accuracy = 100 * correct / total
        val_tolerance_accuracy = 100 * within_tolerance / total
        val_accuracies.append(val_accuracy)
        val_tolerance_accuracies.append(val_tolerance_accuracy)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        scheduler.step()

        elapsed_time = time.time() - start_time
        eta = (elapsed_time / (epoch + 1)) * (num_epochs - epoch - 1)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracy:.2f}%, Val Tolerance Acc: {val_tolerance_accuracy:.2f}% - ETA: {eta/60:.2f} min")

    # Final Training Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()
