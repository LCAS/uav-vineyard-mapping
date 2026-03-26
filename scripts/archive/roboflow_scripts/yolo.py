import torch
import cv2

# Path to your custom YOLO model
model_path = '../../yolo_models/yolo11x.pt'

# Input image path
input_image_path = '../../images/39_feet/DJI_20240802142831_0001_W.JPG'

# Output image path
output_image_path = '../../images/output_image_yolo.png'

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Load the image
image = cv2.imread(input_image_path)
assert image is not None, f"Image not found: {input_image_path}"

# Run YOLO detection
results = model(image)

# Save results
results.save(save_dir='runs/detect/exp')

# Locate the saved image and move it to your output path
annotated_image_path = f'runs/detect/exp/{input_image_path.split("/")[-1]}'
annotated_image = cv2.imread(annotated_image_path)
cv2.imwrite(output_image_path, annotated_image)

print(f"Detection complete. Output saved to: {output_image_path}")
