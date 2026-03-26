import subprocess
import json

# Path to one of your P1 images
IMAGE_PATH = "../../images/agri_tech_centre/jojo/DJI_20250331152402_0103.JPG"
OUTPUT_FILE = "dji_metadata_dump.txt"

def dump_all_metadata():
    print(f"Extracting all metadata from {IMAGE_PATH}...")
    
    # -a (Duplicates) -u (Unknown tags) -g1 (Group by location)
    cmd = ['exiftool', '-a', '-u', '-g1', IMAGE_PATH]
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        result = subprocess.run(cmd, stdout=outfile, text=True)
        
    print(f"Done! Open '{OUTPUT_FILE}' and search for the keywords below.")

dump_all_metadata()