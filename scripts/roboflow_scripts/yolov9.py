import glob
import os
import shutil
from IPython.display import Image, display
import subprocess

# Define paths
weights_path = "/home/cheddar/code/yolov9/runs/train/exp4/weights/best.pt"
#source_path = "/home/cheddar/code/yolov9/data/vineyard/vineyard_test-6/test/images"
source_path = "/home/cheddar/code/vineyard_detection/images/39_feet"
detect_script = "/home/cheddar/code/yolov9/detect.py"
output_dir = "/home/cheddar/code/yolov9/runs/detect/exp4"
save_dir = "/home/cheddar/code/yolov9/saved_detections"

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Run YOLO detection
detect_command = [
    "python", detect_script,
    "--img", "1280",
    "--conf", "0.1",
    "--device", "0",
    "--weights", weights_path,
    "--source", source_path,
    "--save-txt", "--save-conf"  # Ensure detected results are saved
]

print(f"Running detection with the command: {' '.join(detect_command)}")
subprocess.run(detect_command)

# Find detected images
detected_images = glob.glob(os.path.join(output_dir, "*.jpg"))

if detected_images:
    for image_path in detected_images:
        # Copy detected images to the save directory
        shutil.copy(image_path, save_dir)

        # Display images
        display(Image(filename=image_path, width=600))
        print("\n")
    
    print(f"Saved detected images to {save_dir}")
else:
    print("No images found in the detection folder.")
