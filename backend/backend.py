from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import base64
from io import BytesIO
import os
from torchvision import models
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Segmentation Model (DeepLabV3 with ResNet50 backbone)
segmentation_model = torch.load(r'C:\Users\almaa\OneDrive\Desktop\tkr_ai\backend\saved_model\segmentation_model_2.0.pth', map_location=device)
segmentation_model.to(device)
segmentation_model.eval()

# Load Severity Prediction Model (ResNet34)
class SeverityModel(nn.Module):
    def __init__(self):
        super(SeverityModel, self).__init__()
        self.model = models.resnet34(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 5)
        )

    def forward(self, x):
        return self.model(x)

severity_model = SeverityModel()
severity_model.load_state_dict(torch.load(r'C:\Users\almaa\OneDrive\Desktop\tkr_ai\backend\best_oa_severity_model.pth', map_location=device))
severity_model.to(device)
severity_model.eval()

app = Flask(__name__)
CORS(app)

# Preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).unsqueeze(0).to(device)

# Encode colorful mask to base64
def encode_colorful_mask(mask):
    # Generate a color map
    num_classes = int(np.max(mask)) + 1
    cmap = plt.get_cmap('tab20', num_classes)
    rgb_mask = cmap(mask / num_classes)[:, :, :3] * 255  # Convert to RGB
    rgb_mask = Image.fromarray(rgb_mask.astype(np.uint8))
    buffered = BytesIO()
    rgb_mask.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Encode original image to base64
def encode_original_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Feature extraction function for femur and tibia widths
def extract_features(mask):
    femur_id, tibia_id = 1, 2
    features = {}

    def calculate_geometric_widths(mask, label_id, region='full'):
        # Create a binary mask for the label
        binary_mask = (mask == label_id).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0

        # Get bounding box around the contour
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

        # Define proximal and distal regions
        if region == 'proximal':
            region_mask = binary_mask[y:y + h // 3, :]
        elif region == 'distal':
            region_mask = binary_mask[y + 2 * h // 3:y + h, :]
        else:
            region_mask = binary_mask

        # Find widths based on the selected region
        region_contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not region_contours:
            return 0

        max_width = 0
        for contour in region_contours:
            _, _, region_w, _ = cv2.boundingRect(contour)
            max_width = max(max_width, region_w)

        return max_width

    # Femur widths
    features['femur_proximal_width'] = calculate_geometric_widths(mask, femur_id, 'proximal')
    features['femur_distal_width'] = calculate_geometric_widths(mask, femur_id, 'distal')
    features['femur_max_width'] = calculate_geometric_widths(mask, femur_id)

    # Tibia widths
    features['tibia_proximal_width'] = calculate_geometric_widths(mask, tibia_id, 'proximal')
    features['tibia_distal_width'] = calculate_geometric_widths(mask, tibia_id, 'distal')
    features['tibia_max_width'] = calculate_geometric_widths(mask, tibia_id)

    return features

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate image input
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded."}), 400

        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        original_image = encode_original_image(image)
        processed_image = preprocess_image(image)

        # Segmentation model prediction
        with torch.no_grad():
            segmentation_output = segmentation_model(processed_image)['out']
        mask = torch.argmax(segmentation_output, dim=1)[0].cpu().numpy()
        segmented_image = encode_colorful_mask(mask)

        # Extract features
        features = extract_features(mask)

        # Severity prediction model
        with torch.no_grad():
            severity_output = severity_model(processed_image)
        severity_class = int(torch.softmax(severity_output, dim=1).argmax(1).item())
        severity_score = float(torch.softmax(severity_output, dim=1).max(1).values.item())

        # Map severity score to label
        severity_labels = ['Normal', 'Mild', 'Moderate', 'Severe', 'Very Severe']
        severity_label = severity_labels[severity_class]

        # Return results with colorful mask and features
        return jsonify({
            "segmentation": "Segmentation Successful",
            "severity": severity_label,
            "severity_score": severity_class,
            "original_image": original_image,
            "segmented_image": segmented_image,
            "features": features
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
