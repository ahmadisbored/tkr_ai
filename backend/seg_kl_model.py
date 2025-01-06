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
import pandas as pd
import cv2

# Paths
image_dir = './KneeXrayAllImgs/'  # Directory containing all images
output_dir = './auto_segment_output/'
mask_dir = os.path.join(output_dir, 'masks')
visual_dir = os.path.join(output_dir, 'visuals')
features_csv = os.path.join(output_dir, 'features.csv')

os.makedirs(mask_dir, exist_ok=True)
os.makedirs(visual_dir, exist_ok=True)

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset Definition
class KneeDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, img_name

# Load Dataset
dataset = KneeDataset(image_dir, transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load Saved Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('./saved_model/segmentation_model_2.0.pth')
model = model.to(device)
model.eval()

# Constants
NUM_CLASSES = 11

# Function to Calculate Widths
def calculate_widths(mask, category_id):
    rows, cols = np.where(mask == category_id)
    if len(rows) == 0:
        return 0.0
    min_col, max_col = np.min(cols), np.max(cols)
    width_px = max_col - min_col
    return width_px

# Function to Calculate JSW Metrics
def calculate_jsw(mask_femur, mask_tibia):
    # Fix: Use boolean masks directly
    femur_rows, femur_cols = np.where(mask_femur)
    tibia_rows, tibia_cols = np.where(mask_tibia)

    if len(femur_rows) == 0 or len(tibia_rows) == 0:
        return 0.0, 0.0

    jsw_values = []
    for col in range(mask_femur.shape[1]):
        femur_points = femur_rows[femur_cols == col]
        tibia_points = tibia_rows[tibia_cols == col]
        if len(femur_points) > 0 and len(tibia_points) > 0:
            jsw = np.min(tibia_points) - np.max(femur_points)
            if jsw > 0:  # Only count valid joint space
                jsw_values.append(jsw)

    if len(jsw_values) == 0:
        return 0.0, 0.0

    min_jsw_px = np.min(jsw_values)
    avg_jsw_px = np.mean(jsw_values)
    return min_jsw_px, avg_jsw_px

# Process Images and Save Outputs
features = []
with torch.no_grad():
    for images, img_names in dataloader:
        images = images.to(device)
        outputs = model(images)['out']
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        for i in range(images.size(0)):
            img_name = img_names[i]
            pred_mask = preds[i]

            # Calculate widths
            femur_width_px = calculate_widths(pred_mask, 1)
            tibia_width_px = calculate_widths(pred_mask, 2)

            # Calculate JSW metrics
            min_jsw_px, avg_jsw_px = calculate_jsw(pred_mask == 1, pred_mask == 2)

            # Save Features
            features.append([img_name, femur_width_px, tibia_width_px, min_jsw_px, avg_jsw_px])

            # Save mask
            mask_path = os.path.join(mask_dir, img_name.replace('.jpg', '_mask.png'))
            plt.imsave(mask_path, pred_mask, cmap='tab10')

            # Save visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(images[i].permute(1, 2, 0).cpu())
            axes[0].set_title('Original Image')
            axes[1].imshow(pred_mask, cmap='tab10')
            axes[1].set_title('Segmented Image')
            axes[1].text(10, 20, f'Femur Width: {femur_width_px:.2f}px', color='white')
            axes[1].text(10, 40, f'Tibia Width: {tibia_width_px:.2f}px', color='white')
            axes[1].text(10, 60, f'Min JSW: {min_jsw_px:.2f}px', color='white')
            axes[1].text(10, 80, f'Avg JSW: {avg_jsw_px:.2f}px', color='white')
            visual_path = os.path.join(visual_dir, img_name.replace('.jpg', '_visual.png'))
            plt.savefig(visual_path)
            print(f"Saved {img_name}")
            plt.close()

# Save Features to CSV
features_df = pd.DataFrame(features, columns=['Image', 'Femur_Width_px', 'Tibia_Width_px', 'Min_JSW_px', 'Avg_JSW_px'])
features_df.to_csv(features_csv, index=False)

print(f"Processing complete. Masks saved in {mask_dir}, visuals saved in {visual_dir}, and features saved in {features_csv}")
