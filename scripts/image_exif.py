'''
This script utilises ExifTool to extract and display all available metadata from an image file in JSON format. 
It parses the output to print each metadata tag and its corresponding value.
'''

import subprocess
import json

def extract_exif_data(image_path):
    try:
        # Call ExifTool to extract metadata in JSON format
        result = subprocess.run(['exiftool', '-json', image_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return

        # Parse the JSON output from ExifTool
        metadata = json.loads(result.stdout)
        if metadata:
            for tag, value in metadata[0].items():
                print(f"{tag}: {value}")
        else:
            print("No metadata found.")
    except Exception as e:
        print(f"Error: {e}")

image_path = "../images/riseholme/august_2024/39_feet/DJI_20240802142835_0003_W.JPG"
# image_path = "../images/jojo/agri_tech_centre/RX1RII/DSC00610.JPG"
# image_path = "../images/outfields/wraxall/topdown/rgb/DJI_20241004151155_0009_D.JPG"
# image_path="../images/outfields/jojo/topdown/DJI_20240618141121_0011_D.JPG"
extract_exif_data(image_path)
