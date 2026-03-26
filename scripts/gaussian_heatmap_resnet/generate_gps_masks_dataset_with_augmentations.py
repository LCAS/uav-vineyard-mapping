import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A

# ==========================================================
# CONFIGURATION
# ==========================================================
# List all training folders you want to augment
TRAIN_FOLDERS = [
    "dataset_gps_masks/riseholme/full_res/valid",
    "dataset_gps_masks/riseholme/patches_960x1280/valid",
    "dataset_gps_masks/riseholme/patches_480x640/valid"
]

# Base output folder
OUTPUT_BASE = "dataset_gps_masks_with_augmentations/riseholme"

# --- Augmentation parameters ---
AUGMENTATIONS_PER_IMAGE = 3  # Outputs per training example
ROTATION_LIMIT = 15          # Degrees, ± value
SHEAR_X_LIMIT = 15           # Degrees, ± value horizontal
SHEAR_Y_LIMIT = 15           # Degrees, ± value vertical
BRIGHTNESS_LIMIT = 0.2       # ±20%
CONTRAST_LIMIT = 0.0         # No contrast adjustment

# Subfolders (must exist in training folders)
SUBFOLDERS = {
    "images": "images",
    "posts": "posts_masks",
    "rows": "rows_masks",
    "vine": "vine_masks",
}

# ==========================================================
# AUGMENTATION PIPELINE
# ==========================================================
transform = A.Compose([
    A.Affine(
        rotate=(-ROTATION_LIMIT, ROTATION_LIMIT),
        shear={"x": (-SHEAR_X_LIMIT, SHEAR_X_LIMIT), "y": (-SHEAR_Y_LIMIT, SHEAR_Y_LIMIT)},
        fit_output=False,
        p=1.0,
        border_mode=cv2.BORDER_REFLECT_101,  # ✅ fixed
    ),
    A.RandomBrightnessContrast(
        brightness_limit=BRIGHTNESS_LIMIT,
        contrast_limit=CONTRAST_LIMIT,
        p=1.0,
    )
])

# ==========================================================
# UTILITIES
# ==========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def augment_and_save(image_path, posts_path, rows_path, vine_path, out_dirs):
    """Apply augmentations to image + masks and save them."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARN] Could not load {image_path}")
        return

    # Load and resize masks to match image
    masks = {}
    for key, mask_path in zip(["posts", "rows", "vine"], [posts_path, rows_path, vine_path]):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        masks[key] = mask

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for i in range(AUGMENTATIONS_PER_IMAGE):
        augmented = transform(
            image=image,
            masks=[masks["posts"], masks["rows"], masks["vine"]],
        )

        aug_image = augmented["image"]
        aug_posts, aug_rows, aug_vine = augmented["masks"]

        aug_name = f"{base_name}_aug_{i+1}.jpg"

        # Save image and masks
        cv2.imwrite(os.path.join(out_dirs["images"], aug_name), aug_image)
        cv2.imwrite(os.path.join(out_dirs["posts"], f"posts_mask_{aug_name}"), aug_posts)
        cv2.imwrite(os.path.join(out_dirs["rows"], f"rows_mask_{aug_name}"), aug_rows)
        cv2.imwrite(os.path.join(out_dirs["vine"], f"vine_mask_{aug_name}"), aug_vine)

# ==========================================================
# MAIN
# ==========================================================
def main():
    for base_dir in TRAIN_FOLDERS:
        # Compute output folder based on input folder
        relative_path = os.path.relpath(base_dir, "dataset_gps_masks/riseholme")
        output_dir = os.path.join(OUTPUT_BASE, relative_path)

        print(f"🔧 Augmenting dataset in {base_dir}")
        print(f"🔹 Output will be saved to {output_dir}")

        # Prepare output directories
        out_dirs = {}
        for key, sub in SUBFOLDERS.items():
            path = os.path.join(output_dir, sub)
            ensure_dir(path)
            out_dirs[key] = path

        img_dir = os.path.join(base_dir, SUBFOLDERS["images"])
        posts_dir = os.path.join(base_dir, SUBFOLDERS["posts"])
        rows_dir = os.path.join(base_dir, SUBFOLDERS["rows"])
        vine_dir = os.path.join(base_dir, SUBFOLDERS["vine"])

        images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])

        for img_name in tqdm(images, desc=f"Augmenting images in {relative_path}"):
            image_path = os.path.join(img_dir, img_name)
            posts_path = os.path.join(posts_dir, f"posts_mask_{img_name}")
            rows_path = os.path.join(rows_dir, f"rows_mask_{img_name}")
            vine_path = os.path.join(vine_dir, f"vine_mask_{img_name}")

            augment_and_save(image_path, posts_path, rows_path, vine_path, out_dirs)

    print("✅ All augmentations complete!")

if __name__ == "__main__":
    main()
