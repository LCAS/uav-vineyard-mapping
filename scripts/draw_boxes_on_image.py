from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import json

# Load the image
# image_path = "../images/riseholme/august_2024/39_feet/DJI_20240802143112_0076_W.JPG"
image_path = "../images/riseholme/march_2025/39_feet/DJI_20250310145022_0076_W.JPG"
image = Image.open(image_path)
draw = ImageDraw.Draw(image)

# List of bounding boxes and labels
boxes = [
  {"point": [949, 277], "label": "posts"},
  {"point": [864, 670], "label": "posts"},
  {"point": [688, 737], "label": "posts"},
  {"point": [702, 50], "label": "posts"},
  {"point": [514, 150], "label": "posts"},
  {"point": [356, 293], "label": "posts"},
  {"point": [158, 404], "label": "posts"},
  {"point": [25, 70], "label": "posts"},
  {"point": [24, 928], "label": "posts"},
  {"point": [246, 818], "label": "posts"},
  {"point": [486, 727], "label": "posts"},
  {"point": [836, 970], "label": "posts"}
]

# Draw the boxes
for item in boxes:
    x1, y1 = item["point"]
    draw.circle([x1, y1], outline="red", radius=100, width=5)

# Save the modified image (e.g., with "_boxed" suffix)
output_path = "boxed_image.jpg"
image.save(output_path)
print(f"Saved image with boxes to: {output_path}")
