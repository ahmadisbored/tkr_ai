import os
import json
import time
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image
from skimage.draw import polygon
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch import DeepLabV3Plus
from segmentation_models_pytorch import PSPNet

# Paths
json_path = './annotations/instances_default.json'
output_dir = './segtraining/'

# Conversion ratio
conversion_ratio = 0.4  # Pixels to mm

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

# Dataset
class KneeDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_id = next((key for key, value in image_id_to_file.items() if value == img_name), None)
        img_path = os.path.join(self.image_dir, img_name)

        # Normalize path for consistency
        img_path = os.path.normpath(img_path)
        image = Image.open(img_path).convert('RGB')

        # Create multi-class mask
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        if img_id in annotations:
            for ann in annotations[img_id]:
                category_id = ann['category_id']
                if 'segmentation' not in ann or not ann['segmentation']:
                    continue
                segmentation = ann['segmentation'][0]
                if len(segmentation) < 6:
                    continue
                polygon_points = [(int(segmentation[i]), int(segmentation[i + 1])) for i in range(0, len(segmentation), 2)]
                rr, cc = polygon([p[1] for p in polygon_points], [p[0] for p in polygon_points], mask.shape)
                mask[rr, cc] = category_id

        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize((224, 224))(Image.fromarray(mask))
            mask = torch.tensor(np.array(mask), dtype=torch.long)
        return image, mask, img_name

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load Data
train_dataset = KneeDataset(os.path.join(output_dir, 'train'), transform=transform)
val_dataset = KneeDataset(os.path.join(output_dir, 'val'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Model Evaluation
def evaluate_model(model, dataloader, num_classes):
    model.eval()
    total_correct = 0
    total_pixels = 0
    predictions = []
    dice_scores = []
    iou_scores = []
    losses = []
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, masks, _ in dataloader:
            images, masks = images.to(device), masks.to(device)
            # Generate outputs
            if isinstance(model, Unet):
                outputs = model(images)
            else:
                outputs = model(images)

            # Handle outputs for DeepLabV3Plus
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs['out']
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == masks).sum().item()
            total_pixels += masks.numel()
            predictions.append(preds.cpu().numpy())

            # Calculate Dice Score
            intersection = (preds & masks).float().sum((1, 2))
            union = (preds | masks).float().sum((1, 2))
            dice = (2. * intersection / (union + 1e-6)).mean().item()
            dice_scores.append(dice)

            # Calculate IoU Score
            iou = (intersection / (union + 1e-6)).mean().item()
            iou_scores.append(iou)

            # Calculate Loss
            loss = loss_fn(outputs, masks)
            losses.append(loss.item())

    accuracy = total_correct / total_pixels
    avg_loss = np.mean(losses)
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    return predictions, accuracy, avg_loss, avg_dice, avg_iou

# Extract measurements
def extract_measurements(mask):
    # Identify the joint space region
    joint_space_mask = (mask == 3)
    joint_row = np.where(joint_space_mask.any(axis=1))[0]
    if len(joint_row) == 0:
        return 0, 0, 0, 0, 0, 0  # No joint space detected

    joint_space_top = joint_row.min()
    joint_space_bottom = joint_row.max()

    # Femur measurements
    femur_mask = (mask == 1) | (mask == 4) | (mask == 6)
    femur_distal = femur_mask[max(0, joint_space_top - 10):joint_space_top, :].sum(axis=1).max() if joint_space_top > 0 else 0
    femur_proximal = femur_mask[:max(0, joint_space_top - 10), :].sum(axis=1).max() if joint_space_top > 10 else 0
    femur_max = femur_mask.sum(axis=1).max()

    # Tibia measurements
    tibia_mask = (mask == 2) | (mask == 5) | (mask == 7)
    tibia_proximal = tibia_mask[joint_space_bottom:min(mask.shape[0], joint_space_bottom + 10), :].sum(axis=1).max() if joint_space_bottom < mask.shape[0] else 0
    tibia_distal = tibia_mask[min(mask.shape[0], joint_space_bottom + 10):, :].sum(axis=1).max() if joint_space_bottom + 10 < mask.shape[0] else 0
    tibia_max = tibia_mask.sum(axis=1).max()

    return femur_distal, femur_proximal, femur_max, tibia_distal, tibia_proximal, tibia_max

# Compare Models
num_classes = len(categories) + 1
models_to_test = {
    'ResNet50': models.segmentation.deeplabv3_resnet50(weights='DEFAULT'),
    'ResNet101': models.segmentation.deeplabv3_resnet101(weights='DEFAULT'),
    'FCN_ResNet50': models.segmentation.fcn_resnet50(weights='DEFAULT'),
    'FCN_ResNet101': models.segmentation.fcn_resnet101(weights='DEFAULT'),
    'U-Net': Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=num_classes),
    'DeepLabV3+_EfficientNet': DeepLabV3Plus(encoder_name="efficientnet-b7", encoder_weights="imagenet", classes=num_classes),
    'PSPNet': PSPNet(encoder_name="resnet101", encoder_weights="imagenet", classes=num_classes)
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Train and Visualize Results
plt.figure(figsize=(20, 20))
plt.suptitle(f"Measurements in mm (Conversion Ratio: {conversion_ratio} mm/pixel)")

# Plot the original image and mask at the top
sample_image, sample_mask, sample_name = next(iter(val_loader))
plt.subplot(1, len(models_to_test) + 2, 1)
plt.imshow(np.transpose(sample_image[0].numpy(), (1, 2, 0)))
plt.title(f'Original Image\n{sample_name[0]}')
plt.axis('off')

plt.subplot(1, len(models_to_test) + 2, 2)
plt.imshow(sample_mask[0].cpu().numpy())
plt.title('Ground Truth Mask')
plt.axis('off')

for model_idx, (model_name, model) in enumerate(models_to_test.items()):
    print(f"Training {model_name}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training
    start_time = time.time()
    for epoch in range(10):
        model.train()
        for images, masks, _ in tqdm(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            # Generate outputs
            if isinstance(model, Unet):
                outputs = model(images)
            else:
                outputs = model(images)

            # Handle outputs for DeepLabV3Plus
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs['out']

            # Compute loss
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()


    # Evaluation
    predictions, accuracy, avg_loss, avg_dice, avg_iou = evaluate_model(model, val_loader, num_classes)
    end_time = time.time()
    training_time = end_time - start_time

    # Measurements
    femur_distal, femur_proximal, femur_max, tibia_distal, tibia_proximal, tibia_max = extract_measurements(predictions[0][0])

    # Visualization
    plt.subplot(1, len(models_to_test) + 2, model_idx + 3)
    plt.imshow(predictions[0][0])
    plt.title(
        f"{model_name}\nAcc: {accuracy:.2%}\nLoss: {avg_loss:.4f}\nDice: {avg_dice:.4f}\nIoU: {avg_iou:.4f}\n"
        f"Time: {training_time:.2f}s\n"
        # f"Femur Distal: {femur_distal}px ({femur_distal * conversion_ratio:.2f} mm)\n"
        # f"Femur Proximal: {femur_proximal}px ({femur_proximal * conversion_ratio:.2f} mm)\n"
        # f"Femur Max: {femur_max}px ({femur_max * conversion_ratio:.2f} mm)\n"
        # f"Tibia Distal: {tibia_distal}px ({tibia_distal * conversion_ratio:.2f} mm)\n"
        # f"Tibia Proximal: {tibia_proximal}px ({tibia_proximal * conversion_ratio:.2f} mm)\n"
        # f"Tibia Max: {tibia_max}px ({tibia_max * conversion_ratio:.2f} mm)"
    )
    plt.axis('off')

plt.tight_layout()
plt.show()
