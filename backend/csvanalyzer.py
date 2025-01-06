import os
from PIL import Image

# Define directory
train_dir = './OA_mask_train'

# Function to remove 'copy' from filenames
for filename in os.listdir(train_dir):
    if 'copy' in filename:
        new_filename = filename.replace(' copy', '')
        src_path = os.path.join(train_dir, filename)
        dest_path = os.path.join(train_dir, new_filename)
        os.rename(src_path, dest_path)

# Convert PNG images to JPG
for filename in os.listdir(train_dir):
    if filename.endswith('.png'):
        img_path = os.path.join(train_dir, filename)
        img = Image.open(img_path)
        new_filename = os.path.splitext(filename)[0] + ".jpg"
        img.convert('RGB').save(os.path.join(train_dir, new_filename))
        os.remove(img_path)

print("Filenames updated and images converted to JPG successfully in", train_dir)
