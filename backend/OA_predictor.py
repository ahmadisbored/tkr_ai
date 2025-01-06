import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Custom dataset for images and features
class OADataset(Dataset):
    def __init__(self, image_paths, features, labels, transform=None):
        self.image_paths = image_paths
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = plt.imread(self.image_paths[idx])
        if self.transform:
            img = self.transform(img)
        # Extract features and labels
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, features, label

# Load data
train_set = pd.read_csv('./train_set_OA.csv')

# Extract features and labels
X = train_set[['Femur_Width_px', 'Tibia_Width_px', 'Min_JSW_px', 'Avg_JSW_px']]
X['Femur_Tibia_Ratio'] = X['Femur_Width_px'] / X['Tibia_Width_px']
X['JSW_Ratio'] = X['Min_JSW_px'] / X['Avg_JSW_px']
X['Width_Ratio'] = X['Femur_Width_px'] / X['Min_JSW_px']
X['Log_JSW'] = np.log1p(X['Avg_JSW_px'])
y = train_set['label']

# Normalize features
X = (X - X.mean()) / X.std()

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Image processing
image_dir = './KneeXray/normalized_train/'

# Split image paths to match train/val sets
train_image_paths = [f'{image_dir}{img.replace("_train", "")}' for img in train_set['Image'][:len(X_train)]]
val_image_paths = [f'{image_dir}{img.replace("_train", "")}' for img in train_set['Image'][-len(X_val):]]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets and loaders
train_dataset = OADataset(train_image_paths, X_train.values, y_train.values, transform=transform)
val_dataset = OADataset(val_image_paths, X_val.values, y_val.values, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define model
class OAClassifier(nn.Module):
    def __init__(self):
        super(OAClassifier, self).__init__()
        # CNN branch
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()
        self.dropout_cnn = nn.Dropout(0.5)

        # Fully connected branch for features
        self.fc_features = nn.Sequential(
            nn.Linear(8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Combined layers
        self.final_fc = nn.Sequential(
            nn.Linear(2048 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 5)
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, img, features):
        cnn_out = self.cnn(img)
        cnn_out = self.dropout_cnn(cnn_out)
        features_out = self.fc_features(features)
        combined = torch.cat((cnn_out, features_out), dim=1)
        return self.final_fc(combined)

# Initialize model, loss, optimizer
model = OAClassifier()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# Training loop
num_epochs = 50
best_val_acc = 0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for imgs, features, labels in progress_bar:
        imgs, features, labels = imgs.to(device), features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs, features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        progress_bar.set_postfix(loss=loss.item(), acc=100. * correct / total)

    scheduler.step()

    val_loss = 0
    val_correct = 0
    val_total = 0
    model.eval()
    with torch.no_grad():
        for imgs, features, labels in val_loader:
            imgs, features, labels = imgs.to(device), features.to(device), labels.to(device)
            outputs = model(imgs, features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = outputs.max(1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100. * val_correct / val_total
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')

# Final evaluation
print("Training complete.")
