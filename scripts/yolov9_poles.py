import numpy as np
import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO('../weights/roboflow_version_8_weights.pt')

# Get class names
class_names = model.names  # Dictionary mapping class IDs to names

# Ensure "pole" is in the class names
if "pole" not in class_names.values():
    raise ValueError("Class 'pole' not found in model classes.")

# Find the index of "pole"
pole_class_id = [k for k, v in class_names.items() if v == "pole"][0]

# Load image and predict
input_image = '../images/riseholme/august_2024/39_feet/DJI_20240802142846_0008_W.JPG'
output_image_path = '../images/yolo_output/detected_poles_only.jpg'

# Read the image
image = cv2.imread(input_image)

# Perform prediction
results = model.predict(input_image, conf=0.2)

# Draw bounding boxes only for "pole" class
for result in results:
    boxes = result.boxes  # Bounding boxes
    for box in boxes:
        class_id = int(box.cls)  # Get class ID
        class_name = class_names[class_id]  # Get class name

        if class_name == "pole":  # Filter only "pole" class
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to int
            confidence = box.conf[0].item()  # Confidence score

            # Define color for "pole" (e.g., green)
            color = (0, 255, 0)  # Green for pole

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Save the image with detected poles
cv2.imwrite(output_image_path, image)
print(f"Saved detected image to: {output_image_path}")
