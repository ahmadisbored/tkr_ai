import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models.segmentation as models
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon

# Paths
json_path = './annotations/instances_default.json'
output_dir = './segtraining/'

# Load JSON annotations
with open(json_path, 'r') as f:
    data = json.load(f)

# Validate JSON categories
categories = {cat['id']: cat['name'] for cat in data['categories']}
print("Categories in JSON:", categories)

# Create mappings for annotations
annotations = {}
for ann in data['annotations']:
    if ann['category_id'] not in categories:  # Skip invalid categories
        continue
    img_id = ann['image_id']
    if img_id not in annotations:
        annotations[img_id] = []
    annotations[img_id].append(ann)

# Map image IDs to file names
image_id_to_file = {img['id']: img['file_name'].replace('normalized_train/', '') for img in data['images']}

# Ensure filenames in JSON match the directory
missing_files = []
for img_id, filename in image_id_to_file.items():
    if not os.path.exists(os.path.join(output_dir, 'train', filename)) and \
       not os.path.exists(os.path.join(output_dir, 'val', filename)) and \
       not os.path.exists(os.path.join(output_dir, 'test', filename)):
        missing_files.append(filename)

if missing_files:
    print(f"Warning: The following files are listed in JSON but not found in directories: {missing_files}")

# Define Dataset (Multi-Class Mask Dataset)
class KneeDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        assert os.path.exists(image_dir), f"Directory {image_dir} does not exist."
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_id = next((key for key, value in image_id_to_file.items() if value == img_name), None)
        img_path = os.path.join(self.image_dir, img_name)

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Create multi-class mask
        mask = np.zeros((image.height, image.width), dtype=np.uint8)

        if img_id in annotations:
            for ann in annotations[img_id]:
                category_id = ann['category_id']  # Use category ID for segmentation
                if 'segmentation' not in ann or not ann['segmentation']:
                    continue
                segmentation = ann['segmentation'][0]
                if len(segmentation) < 6:  # Polygon requires at least 3 points
                    continue
                polygon_points = [(int(segmentation[i]), int(segmentation[i + 1])) for i in range(0, len(segmentation), 2)]

                # Fill polygon directly in mask
                rr, cc = polygon([p[1] for p in polygon_points], [p[0] for p in polygon_points], mask.shape)
                mask[rr, cc] = category_id

        # Resize mask and apply transforms
        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize((224, 224))(Image.fromarray(mask))
            mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Prepare Data Loaders
train_dataset = KneeDataset(os.path.join(output_dir, 'train'), transform=transform)
val_dataset = KneeDataset(os.path.join(output_dir, 'val'), transform=transform)
test_dataset = KneeDataset(os.path.join(output_dir, 'test'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Load Model
NUM_CLASSES = 11
model = models.deeplabv3_resnet50(weights=models.DeepLabV3_ResNet50_Weights.DEFAULT)
model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=(1, 1))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss Function
loss_fn = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Function to calculate femur and tibia widths
PIXEL_TO_MM = 0.368  # Example ratio (adjust as needed)

def calculate_widths(mask, category_id):
    rows, cols = np.where(mask == category_id)
    if len(rows) == 0:
        return 0.0, 0.0
    min_col, max_col = np.min(cols), np.max(cols)
    width_px = max_col - min_col
    width_mm = width_px * PIXEL_TO_MM
    return width_px, width_mm

# Evaluation Metrics
def evaluate_model(model, dataloader):
    model.eval()
    total_loss, total_iou, total_correct, total_pixels = 0, 0, 0, 0
    class_accuracies = {c: {'TP': 0, 'FP': 0, 'FN': 0} for c in range(NUM_CLASSES)}

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == masks).sum().item()
            total_pixels += masks.numel()

            for c in range(NUM_CLASSES):
                preds_c = preds == c
                targets_c = masks == c
                intersection = (preds_c & targets_c).sum().item()
                union = (preds_c | targets_c).sum().item()
                iou = intersection / (union + 1e-6)
                total_iou += iou

    accuracy = (total_correct / total_pixels) * 100
    return total_loss / len(dataloader), total_iou / (len(dataloader) * NUM_CLASSES), accuracy

# Train Model
def train_model(model, train_loader, val_loader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            loss = loss_fn(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss, iou, accuracy = evaluate_model(model, val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, IoU: {iou:.4f}, Accuracy: {accuracy:.2f}%")

# Execute Training and Visualization
num_epochs = 50
train_model(model, train_loader, val_loader, num_epochs)

# Evaluate on Test Data
test_loss, test_iou, test_accuracy = evaluate_model(model, test_loader)
print(f"Test Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Final Visualization
def visualize_predictions(model, dataloader):
    model.eval()
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)

            for i in range(images.size(0)):
                # Calculate femur and tibia widths - Highlighted Code Start
                femur_width_px, femur_width_mm = calculate_widths(preds[i].cpu().numpy(), 1)
                tibia_width_px, tibia_width_mm = calculate_widths(preds[i].cpu().numpy(), 2)
                # Highlighted Code End

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(images[i].permute(1, 2, 0).cpu())
                axes[0].set_title('X-ray')
                axes[1].imshow(masks[i].cpu(), cmap='tab10')
                axes[1].set_title('Mask')
                axes[2].imshow(preds[i].cpu(), cmap='tab10')
                axes[2].set_title('Prediction')
                axes[2].text(10, 20, f'Femur Width: {femur_width_px:.2f}px ({femur_width_mm:.2f} mm)', color='white')
                axes[2].text(10, 40, f'Tibia Width: {tibia_width_px:.2f}px ({tibia_width_mm:.2f} mm)', color='white')
                plt.show()
            break

visualize_predictions(model, test_loader)

# Save Model Architecture and Weights
MODEL_PATH = './saved_model'
os.makedirs(MODEL_PATH, exist_ok=True)

# Save entire model architecture and weights
torch.save(model, os.path.join(MODEL_PATH, 'segmentation_model_2.0.pth'))
