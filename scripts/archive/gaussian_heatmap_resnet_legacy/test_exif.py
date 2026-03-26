#!/usr/bin/env python3
import json
import subprocess
import os

image_folder = "../../images/jojo/agri_tech_centre/RX1RII"
script_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(script_dir, image_folder)

images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg'))][:1]

for img_name in images:
    img_path = os.path.join(image_folder, img_name)
    print(f"Testing {img_name}")
    print(f"Full path: {img_path}")
    
    cmd = ['exiftool', '-json', '-n', img_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        meta = json.loads(result.stdout)[0]
        print(f"Yaw: {meta.get('Yaw')}")
        print(f"Pitch: {meta.get('Pitch')}")
        print(f"Roll: {meta.get('Roll')}")
        print(f"SonyImageWidth: {meta.get('SonyImageWidth')}")
        print(f"SonyImageHeight: {meta.get('SonyImageHeight')}")
        print(f"ImageWidth: {meta.get('ImageWidth')}")
        print(f"ImageHeight: {meta.get('ImageHeight')}")
        print(f"GPSLatitude: {meta.get('GPSLatitude')}")
        print(f"GPSLongitude: {meta.get('GPSLongitude')}")
        print(f"GPSAltitude: {meta.get('GPSAltitude')}")
    else:
        print("No EXIF data found")
