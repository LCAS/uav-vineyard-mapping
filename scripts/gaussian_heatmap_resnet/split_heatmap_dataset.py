import os
import random
import shutil
from pathlib import Path

# Input directories
image_dirs = [
    "../../images/riseholme/august_2024/39_feet",
    "../../images/riseholme/august_2024/65_feet",
    "../../images/riseholme/august_2024/100_feet",
    "../../images/riseholme/march_2025/39_feet",
    "../../images/riseholme/march_2025/65_feet",
    "../../images/riseholme/march_2025/100_feet"
]

mask_dirs = [
    "./heatmap_masks/riseholme/august_2024/39_feet",
    "./heatmap_masks/riseholme/august_2024/65_feet",
    "./heatmap_masks/riseholme/august_2024/100_feet",
    "./heatmap_masks/riseholme/march_2025/39_feet",
    "./heatmap_masks/riseholme/march_2025/65_feet",
    "./heatmap_masks/riseholme/march_2025/100_feet"
]

# Output directory
output_root = Path("heatmap_masks_from_yolo_labels_and_gps_labels/vineyard_segmentation_16")

# Define subfolders
splits = ["train", "valid", "test"]
mask_types = {"posts_mask_": "pole", "rows_mask_": "vine_row", "vine_mask_": "trunk"}

for split in splits:
    for subfolder in ["images", "masks"]:
        if subfolder == "masks":
            for mask_type in mask_types.values():
                (output_root / split / subfolder / mask_type).mkdir(parents=True, exist_ok=True)
        else:
            (output_root / split / subfolder).mkdir(parents=True, exist_ok=True)


# Collect all image filenames
all_images = []
for image_dir in image_dirs:
    image_dir = Path(image_dir)
    all_images.extend(list(image_dir.glob("*.JPG")))

print(f"Found {len(all_images)} total images")

# Shuffle for random splitting
random.shuffle(all_images)

# Split ratios
n_total = len(all_images)
n_train = int(n_total * 0.7)
n_valid = int(n_total * 0.2)
n_test = n_total - n_train - n_valid

splits_data = {
    "train": all_images[:n_train],
    "valid": all_images[n_train:n_train + n_valid],
    "test": all_images[n_train + n_valid:]
}


def find_and_copy_masks(image_path: Path, split_name: str):
    """Find corresponding masks for an image and copy them to correct location."""
    image_name = image_path.name
    image_stem = image_path.stem  # e.g., DJI_20240802142831_0001_W

    # Copy image
    dest_img_path = output_root / split_name / "images" / image_name
    shutil.copy(image_path, dest_img_path)

    # Find masks
    for mask_dir in mask_dirs:
        for prefix, mask_type in mask_types.items():
            mask_path = Path(mask_dir) / f"{prefix}{image_name}"
            if mask_path.exists():
                new_mask_name = f"{image_stem}_mask.JPG"
                dest_mask_path = output_root / split_name / "masks" / mask_type / new_mask_name
                shutil.copy(mask_path, dest_mask_path)


# Process all splits
for split_name, images in splits_data.items():
    print(f"Processing {split_name}: {len(images)} images")
    for img_path in images:
        find_and_copy_masks(img_path, split_name)

print("✅ Dataset successfully organized and split.")
