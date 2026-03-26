import os
from PIL import Image
from tqdm import tqdm

# ======================
# CONFIGURATION
# ======================
DATASET_ROOT = "heatmap_masks_from_yolo_labels/vineyard_segmentation_16"
OUTPUT_ROOT = "heatmap_masks_from_yolo_labels/patch_images/vineyard_segmentation_16_patch_03_overlap_800px"

PATCH_WIDTH = 800        # desired output width in pixels
OVERLAP = 0.3            # 10% overlap
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

SUBSETS = ["train", "val", "test"]
MASK_CLASSES = ["pole", "trunk", "vine_row"]
INCLUDE_OVERLAY = True

os.makedirs(OUTPUT_ROOT, exist_ok=True)


def find_corresponding_file(base_name, folder, suffix):
    """Find a file that matches a pattern like base_name + suffix (e.g. _mask, _overlay)."""
    for f in os.listdir(folder):
        if f.startswith(base_name) and suffix in f:
            return os.path.join(folder, f)
    return None


def split_image_and_masks(image_path, masks_dirs, overlay_dir, output_dir):
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Load image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Scale width to PATCH_WIDTH, keep ratio
    scale = PATCH_WIDTH / width
    new_height = int(height * scale)
    image = image.resize((PATCH_WIDTH, new_height), Image.Resampling.LANCZOS)

    # Compute stride
    stride = int(PATCH_WIDTH * (1 - OVERLAP))

    # Create output folders
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    if INCLUDE_OVERLAY:
        os.makedirs(os.path.join(output_dir, "overlay"), exist_ok=True)
    for cls in MASK_CLASSES:
        os.makedirs(os.path.join(output_dir, "masks", cls), exist_ok=True)

    # Horizontal tiling
    num_patches = max(1, (width - PATCH_WIDTH) // stride + 1)
    for i in range(num_patches):
        left = i * stride
        right = min(left + PATCH_WIDTH, width)
        if right - left < PATCH_WIDTH:
            left = right - PATCH_WIDTH
        box = (left, 0, right, height)

        # Crop and save image
        img_patch = image.crop((0, 0, PATCH_WIDTH, new_height))
        img_patch_name = f"{base_name}_patch_{i}.png"
        img_patch.save(os.path.join(output_dir, "images", img_patch_name))

        # Process masks
        for cls in MASK_CLASSES:
            mask_dir = masks_dirs[cls]
            mask_path = find_corresponding_file(base_name, mask_dir, "_mask")
            if mask_path:
                mask = Image.open(mask_path).resize((PATCH_WIDTH, new_height), Image.Resampling.NEAREST)
                mask_patch = mask.crop((0, 0, PATCH_WIDTH, new_height))
                mask_patch.save(os.path.join(output_dir, "masks", cls, img_patch_name))
            else:
                print(f"⚠️ Missing {cls} mask for {base_name}")

        # Process overlay
        if INCLUDE_OVERLAY and overlay_dir:
            overlay_path = find_corresponding_file(base_name, overlay_dir, "_overlay")
            if overlay_path:
                overlay = Image.open(overlay_path).resize((PATCH_WIDTH, new_height), Image.Resampling.LANCZOS)
                overlay_patch = overlay.crop((0, 0, PATCH_WIDTH, new_height))
                overlay_patch.save(os.path.join(output_dir, "overlay", img_patch_name))
            else:
                print(f"⚠️ Missing overlay for {base_name}")


# ======================
# MAIN LOOP
# ======================
for subset in SUBSETS:
    print(f"\n🔹 Processing subset: {subset}")

    images_dir = os.path.join(DATASET_ROOT, subset, "images")
    overlay_dir = os.path.join(DATASET_ROOT, subset, "overlay") if INCLUDE_OVERLAY else None
    masks_dirs = {cls: os.path.join(DATASET_ROOT, subset, "masks", cls) for cls in MASK_CLASSES}
    output_dir = os.path.join(OUTPUT_ROOT, subset)

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]

    for img_file in tqdm(image_files, desc=f"Splitting {subset}"):
        image_path = os.path.join(images_dir, img_file)
        split_image_and_masks(image_path, masks_dirs, overlay_dir, output_dir)

print("\n✅ Done! Split dataset saved to:")
print(OUTPUT_ROOT)
