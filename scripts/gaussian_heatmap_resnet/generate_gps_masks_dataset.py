import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil
import pandas as pd

# ==========================================================
# CONFIGURATION
# ==========================================================
IMAGE_DIRS = [
    "../../images/agri_tech_centre/arun_1"
]

MASK_BASE = "terrain_aware_mask_gen/agri_tech_centre/arun_1"
OUTPUT_BASE = "dataset_terrain_aware_mask_gen/agri_tech_centre/arun_1"

PATCH_SIZES = [(960, 1280), (480, 640)] # (height, width)
OVERLAP = 0.2  # 20% overlap

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.1
TEST_SPLIT = 0.2

MASK_TYPES = ["posts", "rows", "vine"]


# ==========================================================
# UTILITIES
# ==========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_all_images():
    images = []
    for img_dir in IMAGE_DIRS:
        for f in os.listdir(img_dir):
            if f.lower().endswith(".jpg"):
                images.append(os.path.join(img_dir, f))
    return sorted(images)


def split_dataset(images):
    train_imgs, test_imgs = train_test_split(images, test_size=TEST_SPLIT, random_state=42)
    train_imgs, val_imgs = train_test_split(train_imgs, test_size=VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT), random_state=42)
    return train_imgs, val_imgs, test_imgs


def generate_manifest(base_dir, subsets=["train", "valid", "test"]):
    """Create CSV manifest mapping images to mask paths for all subsets."""
    records = []
    for subset in subsets:
        img_dir = os.path.join(base_dir, subset, "images")
        posts_dir = os.path.join(base_dir, subset, "posts_masks")
        rows_dir = os.path.join(base_dir, subset, "rows_masks")
        vine_dir = os.path.join(base_dir, subset, "vine_masks")

        if not os.path.exists(img_dir):
            continue

        for f in sorted(os.listdir(img_dir)):
            if not f.lower().endswith(".jpg"):
                continue
            img_path = os.path.join(img_dir, f)
            records.append({
                "subset": subset,
                "image_path": img_path,
                "posts_mask_path": os.path.join(posts_dir, f.replace("DJI_", "posts_mask_DJI_")),
                "rows_mask_path": os.path.join(rows_dir, f.replace("DJI_", "rows_mask_DJI_")),
                "vine_mask_path": os.path.join(vine_dir, f.replace("DJI_", "vine_mask_DJI_"))
            })

    if records:
        df = pd.DataFrame(records)
        manifest_path = os.path.join(base_dir, "manifest.csv")
        df.to_csv(manifest_path, index=False)
        print(f"✅ Manifest created: {manifest_path}")
    else:
        print(f"[WARN] No records found for manifest in {base_dir}")


# ==========================================================
# FULL RESOLUTION COPY
# ==========================================================
def copy_full_res(images, subset):
    """Copy full-resolution images and masks into structured folders."""
    base_dir = os.path.join(OUTPUT_BASE, "full_res", subset)
    img_dir = os.path.join(base_dir, "images")
    posts_dir = os.path.join(base_dir, "posts_masks")
    rows_dir = os.path.join(base_dir, "rows_masks")
    vine_dir = os.path.join(base_dir, "vine_masks")

    for d in [img_dir, posts_dir, rows_dir, vine_dir]:
        ensure_dir(d)

    for img_path in tqdm(images, desc=f"Copying {subset} full-res"):
        img_name = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(img_dir, img_name))

        # Copy associated masks
        for mask_type, out_dir in zip(MASK_TYPES, [posts_dir, rows_dir, vine_dir]):
            # Try multiple naming conventions and extensions in both mask directories
            base_name = os.path.basename(img_path)
            
            # Try different patterns (both mask directory locations)
            patterns = [
                img_path.replace("../../images", "terrain_aware_mask_gen/heatmap_masks").replace("DJI_", f"{mask_type}_mask_DJI_"),
                img_path.replace("../../images", "terrain_aware_mask_gen/heatmap_masks").replace("DJI_", f"{mask_type}_DJI_"),
                img_path.replace("../../images", "terrain_aware_mask_gen/heatmap_masks").replace("DJI_", f"{mask_type}_mask_DJI_").replace(".JPG", ".png"),
                img_path.replace("../../images", "terrain_aware_mask_gen/heatmap_masks").replace("DJI_", f"{mask_type}_DJI_").replace(".JPG", ".png"),
                img_path.replace("../../images", "heatmap_masks").replace("DJI_", f"{mask_type}_mask_DJI_"),
                img_path.replace("../../images", "heatmap_masks").replace("DJI_", f"{mask_type}_DJI_"),
                img_path.replace("../../images", "heatmap_masks").replace("DJI_", f"{mask_type}_mask_DJI_").replace(".JPG", ".png"),
                img_path.replace("../../images", "heatmap_masks").replace("DJI_", f"{mask_type}_DJI_").replace(".JPG", ".png"),
            ]
            
            mask_found = False
            for mask_path in patterns:
                if os.path.exists(mask_path):
                    shutil.copy(mask_path, os.path.join(out_dir, os.path.basename(mask_path)))
                    mask_found = True
                    break
            
            if not mask_found:
                print(f"[WARN] Missing mask for {mask_type}: {base_name}")


# ==========================================================
# PATCH TILING (no black padding, full coverage)
# ==========================================================
def tile_image(img, patch_size, overlap):
    """Split an image into fixed-size patches (no black padding)."""
    h, w = img.shape[:2]
    ph, pw = patch_size
    stride_y = int(ph * (1 - overlap))
    stride_x = int(pw * (1 - overlap))

    patches = []
    y_positions = list(range(0, h - ph + 1, stride_y))
    x_positions = list(range(0, w - pw + 1, stride_x))

    # Include bottom and right edges
    if y_positions[-1] != h - ph:
        y_positions.append(h - ph)
    if x_positions[-1] != w - pw:
        x_positions.append(w - pw)

    for y in y_positions:
        for x in x_positions:
            patch = img[y:y+ph, x:x+pw]
            patches.append((x, y, patch))
    return patches


def create_patched_dataset(images, patch_size, subset):
    """Create tiled patches dataset (images + masks) with structured folders."""
    base_dir = os.path.join(OUTPUT_BASE, f"patches_{patch_size[0]}x{patch_size[1]}", subset)
    img_dir = os.path.join(base_dir, "images")
    posts_dir = os.path.join(base_dir, "posts_masks")
    rows_dir = os.path.join(base_dir, "rows_masks")
    vine_dir = os.path.join(base_dir, "vine_masks")

    for d in [img_dir, posts_dir, rows_dir, vine_dir]:
        ensure_dir(d)

    for img_path in tqdm(images, desc=f"Tiling {subset} {patch_size}"):
        img_name = os.path.basename(img_path).replace(".JPG", "")
        image = cv2.imread(img_path)
        if image is None:
            print(f"[ERROR] Could not load image: {img_path}")
            continue

        patches = tile_image(image, patch_size, OVERLAP)

        for x, y, patch in patches:
            patch_name = f"{img_name}_patch_{x}_{y}.JPG"
            cv2.imwrite(os.path.join(img_dir, patch_name), patch)

            # Save masks
            for mask_type, out_dir in zip(MASK_TYPES, [posts_dir, rows_dir, vine_dir]):
                # Try multiple naming conventions and extensions in both mask directories
                patterns = [
                    img_path.replace("../../images", "terrain_aware_mask_gen/heatmap_masks").replace("DJI_", f"{mask_type}_mask_DJI_"),
                    img_path.replace("../../images", "terrain_aware_mask_gen/heatmap_masks").replace("DJI_", f"{mask_type}_DJI_"),
                    img_path.replace("../../images", "terrain_aware_mask_gen/heatmap_masks").replace("DJI_", f"{mask_type}_mask_DJI_").replace(".JPG", ".png"),
                    img_path.replace("../../images", "terrain_aware_mask_gen/heatmap_masks").replace("DJI_", f"{mask_type}_DJI_").replace(".JPG", ".png"),
                    img_path.replace("../../images", "heatmap_masks").replace("DJI_", f"{mask_type}_mask_DJI_"),
                    img_path.replace("../../images", "heatmap_masks").replace("DJI_", f"{mask_type}_DJI_"),
                    img_path.replace("../../images", "heatmap_masks").replace("DJI_", f"{mask_type}_mask_DJI_").replace(".JPG", ".png"),
                    img_path.replace("../../images", "heatmap_masks").replace("DJI_", f"{mask_type}_DJI_").replace(".JPG", ".png"),
                ]
                
                mask_path = None
                for p in patterns:
                    if os.path.exists(p):
                        mask_path = p
                        break
                
                if mask_path and os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        continue
                    mask_patch = mask[y:y+patch_size[0], x:x+patch_size[1]]
                    mask_name = f"{mask_type}_mask_{img_name}_patch_{x}_{y}.JPG"
                    cv2.imwrite(os.path.join(out_dir, mask_name), mask_patch)


# ==========================================================
# MAIN
# ==========================================================
def main():
    images = get_all_images()
    print(f"Found {len(images)} total images.")

    train_imgs, val_imgs, test_imgs = split_dataset(images)
    print(f"Train: {len(train_imgs)}, Valid: {len(val_imgs)}, Test: {len(test_imgs)}")

    # --- Full resolution ---
    for subset, imgs in zip(["train", "valid", "test"], [train_imgs, val_imgs, test_imgs]):
        copy_full_res(imgs, subset)
    generate_manifest(os.path.join(OUTPUT_BASE, "full_res"))

    # --- Patched datasets ---
    for patch_size in PATCH_SIZES:
        for subset, imgs in zip(["train", "valid", "test"], [train_imgs, val_imgs, test_imgs]):
            create_patched_dataset(imgs, patch_size, subset)
        generate_manifest(os.path.join(OUTPUT_BASE, f"patches_{patch_size[0]}x{patch_size[1]}"))


if __name__ == "__main__":
    main()
