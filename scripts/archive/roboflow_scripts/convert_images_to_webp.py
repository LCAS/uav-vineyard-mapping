import os
import io
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
# SOURCE_FOLDER = "../../images/agri_tech_centre/jojo"  
# OUTPUT_FOLDER = "../../images/agri_tech_centre/webp/jojo"

SOURCE_FOLDER = "../../images/agri_tech_centre/jojo"  
OUTPUT_FOLDER = "../../images/agri_tech_centre/webp/jojo"

MAX_FILE_SIZE_MB = 19.5
# ---------------------

def compress_to_target_size(source_path, output_path, target_mb):
    target_bytes = target_mb * 1024 * 1024
    
    try:
        with Image.open(source_path) as img:
            # Convert to RGB (WebP doesn't support CMYK)
            img = img.convert("RGB")
            
            # Start with high quality
            quality = 90
            min_quality = 50 
            
            while quality >= min_quality:
                # Save to memory buffer first to check size
                buffer = io.BytesIO()
                img.save(buffer, format="WEBP", quality=quality, method=6)
                size = buffer.tell()
                
                if size <= target_bytes:
                    # It fits! Write to disk
                    with open(output_path, "wb") as f:
                        f.write(buffer.getbuffer())
                    
                    # Optional: Comment this out if you don't want to see every single file listed
                    tqdm.write(f"[OK] {os.path.basename(source_path)} | Q:{quality} | {size/1024/1024:.2f} MB")
                    return
                
                # If too big, lower quality and try again
                quality -= 5
            
            tqdm.write(f"[WARNING] Could not compress {os.path.basename(source_path)} under {target_mb}MB.")
            
    except Exception as e:
        tqdm.write(f"[ERROR] Failed to process {source_path}: {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Gather all images first
    print("Scanning folder for images...")
    files = [f for f in os.listdir(SOURCE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not files:
        print("No images found in the source folder.")
    else:
        # The Progress Bar Loop
        for filename in tqdm(files, desc="Processing Images", unit="img"):
            source_file = os.path.join(SOURCE_FOLDER, filename)
            
            # Change extension to .webp
            new_filename = os.path.splitext(filename)[0] + ".jpg"
            output_file = os.path.join(OUTPUT_FOLDER, new_filename)
            
            compress_to_target_size(source_file, output_file, MAX_FILE_SIZE_MB)

        print("\nBatch processing complete!")