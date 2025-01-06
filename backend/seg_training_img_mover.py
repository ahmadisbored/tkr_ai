import os
import shutil
import json
import random
from sklearn.model_selection import train_test_split

# Paths
json_path = './annotations/instances_default.json'  # Path to JSON annotation file
source_dir = './KneeXray/normalized_train'  # Path to original images
output_dir = './segtraining/'  # Path for training, val, and test folders

# Create directories for train, val, and test
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

# Load JSON annotations
with open(json_path, 'r') as f:
    data = json.load(f)

# Extract image file names
images = [img['file_name'].split('/')[-1] for img in data['images']]

# Shuffle and split data
random.seed(42)  # Ensure reproducibility
train_images, test_val_images = train_test_split(images, test_size=0.33, random_state=42)
val_images, test_images = train_test_split(test_val_images, test_size=0.5, random_state=42)

# Function to copy images to respective folders
def copy_images(image_list, dest_folder):
    for image_name in image_list:
        source_path = os.path.join(source_dir, image_name)
        dest_path = os.path.join(dest_folder, image_name)
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)
        else:
            print(f"Warning: {source_path} not found!")

# Copy images to respective directories
copy_images(train_images, os.path.join(output_dir, 'train'))
copy_images(val_images, os.path.join(output_dir, 'val'))
copy_images(test_images, os.path.join(output_dir, 'test'))

print("Dataset split into train, val, and test successfully.")