import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLO model
model = YOLO('../weights/vine_row_segmentation/best.pt')

# Load image and predict
# input_image = '../images/orthophoto/odm_orthophoto-3.png'
# input_image = '../images/riseholme/august_2024/100_feet/DJI_20240802142143_0014_W.JPG'
input_image = '../images/riseholme/august_2024/39_feet/DJI_20240802142942_0034_W.JPG'
# input_image = '../images/wraxall/DJI_20241004151307_0045_D.JPG'
results = model.predict(input_image, conf=0.4)

# Open original image
img = Image.open(input_image).convert("RGB")
width, height = img.size

# Create an empty mask (black image)
mask_image = np.zeros((height, width), dtype=np.uint8)

# Process results
for result in results:
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()  # Convert masks to numpy array
        for mask in masks:
            mask = (mask * 255).astype(np.uint8)  # Scale mask to 0-255
            mask_pil = Image.fromarray(mask).convert("L")
            mask_pil = mask_pil.resize((width, height))  # Resize to match input image size

            # Merge masks using numpy
            mask_image = np.maximum(mask_image, np.array(mask_pil))

# Convert mask to RGB red overlay
mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)  # Black image
mask_rgb[..., 0] = mask_image  # Red channel

# Convert to PIL Image
mask_overlay = Image.fromarray(mask_rgb)

# Blend images (alpha = 0.5 for transparency)
blended = Image.blend(img, mask_overlay, alpha=0.5)

# Save the final overlay
output_path = "../images/segmentation_overlay.png"
blended.save(output_path)

print(f"Saved segmentation overlay to {output_path}")
