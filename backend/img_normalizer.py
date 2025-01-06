import os
from PIL import Image
import numpy as np

# Define the input directory and output directory
input_dir = "./KneeXray/test"
output_dir = "./KneeXray/normalized_test"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through all the .jpg files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        # Full path to the image
        image_path = os.path.join(input_dir, filename)
        
        # Open the image
        image = Image.open(image_path)
        
        # Convert to a NumPy array
        image_array = np.array(image)

        # Check if pixel values are in the range of 0-255
        if np.max(image_array) > 1:
            # Normalize the pixel values to 0-1
            image_array = image_array / 255.0
        
        # Convert back to image format (as saving in 0-1 float range is not directly supported for JPG)
        image_array_normalized = (image_array * 255).astype(np.uint8)
        image_normalized = Image.fromarray(image_array_normalized)
        
        # Save the normalized image to the output directory
        output_path = os.path.join(output_dir, filename)
        image_normalized.save(output_path)

        print(f"Processed and saved: {filename}")

print("All images have been processed and normalized.")
