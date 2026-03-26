import os
import cv2
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon
import albumentations as A

# ==========================================================
# CONFIGURATION
# ==========================================================
DATASETS = [
    os.path.join("dataset_yolov11_labels", "full_res", "train"),
    os.path.join("dataset_yolov11_labels", "patches_960x1280", "train"),
    os.path.join("dataset_yolov11_labels", "patches_480x640", "train"),
]

OUTPUT_BASE = "dataset_yolov11_labels_with_augmentations"

AUGMENTATIONS_PER_IMAGE = 3
ROTATION_LIMIT = 15
SHEAR_X_LIMIT = 15
SHEAR_Y_LIMIT = 15
BRIGHTNESS_LIMIT = 0.2
CONTRAST_LIMIT = 0.0

# ==========================================================
# AUGMENTATION PIPELINE
# ==========================================================
transform = A.Compose([
    A.Affine(
        rotate=(-ROTATION_LIMIT, ROTATION_LIMIT),
        shear={"x": (-SHEAR_X_LIMIT, SHEAR_X_LIMIT), "y": (-SHEAR_Y_LIMIT, SHEAR_Y_LIMIT)},
        fit_output=False,
        p=1.0,
        border_mode=cv2.BORDER_REFLECT_101,
    ),
    A.RandomBrightnessContrast(
        brightness_limit=BRIGHTNESS_LIMIT,
        contrast_limit=CONTRAST_LIMIT,
        p=1.0,
    )
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

# ==========================================================
# UTILITIES
# ==========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def read_yolo_label(label_path, img_shape):
    """Read YOLO polygon label and return list of keypoints [(x, y), ...]."""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    keypoints = []
    classes = []
    h, w = img_shape[:2]
    for line in lines:
        parts = line.strip().split()
        cls = int(parts[0])
        coords = list(map(float, parts[1:]))
        pts = [(coords[i]*w, coords[i+1]*h) for i in range(0, len(coords), 2)]
        keypoints.append(pts)
        classes.append(cls)
    return keypoints, classes

def save_yolo_label(label_path, keypoints_list, classes, img_shape):
    """Save keypoints list back to YOLO format (normalized)."""
    h, w = img_shape[:2]
    with open(label_path, 'w') as f:
        for cls, pts in zip(classes, keypoints_list):
            line = f"{cls} " + " ".join([f"{x/w:.6f} {y/h:.6f}" for x, y in pts]) + "\n"
            f.write(line)

def draw_yolo_polygons(image, keypoints_list, classes):
    """Draw YOLO polygons on the image."""
    img_copy = image.copy()
    for cls, pts in zip(classes, keypoints_list):
        if len(pts) < 3:
            continue
        pts_array = np.array(pts, np.int32).reshape((-1, 1, 2))
        color = tuple(int(x) for x in np.random.randint(50, 255, 3))
        cv2.polylines(img_copy, [pts_array], isClosed=True, color=color, thickness=2)
        text_pos = tuple(pts_array[0][0])
        cv2.putText(img_copy, f"Class {cls}", (text_pos[0], max(20, text_pos[1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img_copy

# ==========================================================
# MAIN AUGMENTATION
# ==========================================================
def main():
    for dataset_dir in DATASETS:
        print(f"Augmenting dataset: {dataset_dir}")
        img_dir = os.path.join(dataset_dir, "images")
        lbl_dir = os.path.join(dataset_dir, "labels")

        relative_path = os.path.relpath(dataset_dir, "dataset_yolov11_labels")
        output_img_dir = os.path.join(OUTPUT_BASE, relative_path, "images")
        output_lbl_dir = os.path.join(OUTPUT_BASE, relative_path, "labels")
        output_ann_dir = os.path.join(OUTPUT_BASE, relative_path, "annotated_images")
        ensure_dir(output_img_dir)
        ensure_dir(output_lbl_dir)
        ensure_dir(output_ann_dir)

        images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])
        for img_name in tqdm(images):
            img_path = os.path.join(img_dir, img_name)
            label_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + ".txt")

            image = cv2.imread(img_path)
            if image is None:
                continue

            keypoints_list, classes = read_yolo_label(label_path, image.shape)
            h, w = image.shape[:2]

            for aug_idx in range(AUGMENTATIONS_PER_IMAGE):
                # Flatten polygons
                flat_kpts = [kp for poly in keypoints_list for kp in poly]

                # Apply Albumentations transform
                augmented = transform(image=image, keypoints=flat_kpts)
                aug_image = augmented["image"]
                aug_kpts_flat = augmented["keypoints"]

                # Rebuild per-polygon keypoints
                aug_keypoints_list = []
                idx = 0
                for poly in keypoints_list:
                    n = len(poly)
                    aug_poly = aug_kpts_flat[idx:idx+n]
                    idx += n
                    # Clip using shapely
                    poly_shape = Polygon(aug_poly)
                    poly_shape = poly_shape.intersection(Polygon([(0,0),(w,0),(w,h),(0,h)]))
                    if poly_shape.is_empty:
                        aug_keypoints_list.append([])
                    else:
                        aug_keypoints_list.append(list(poly_shape.exterior.coords)[:-1])  # remove closing point

                # Save augmented image
                aug_name = f"{os.path.splitext(img_name)[0]}_aug_{aug_idx+1}.jpg"
                cv2.imwrite(os.path.join(output_img_dir, aug_name), aug_image)

                # Save augmented labels
                aug_label_path = os.path.join(output_lbl_dir, os.path.splitext(aug_name)[0]+".txt")
                save_yolo_label(aug_label_path, aug_keypoints_list, classes, aug_image.shape)

                # Save annotated image
                aug_annotated = draw_yolo_polygons(aug_image, aug_keypoints_list, classes)
                cv2.imwrite(os.path.join(output_ann_dir, aug_name), aug_annotated)

    print("✅ Augmentations complete!")

if __name__ == "__main__":
    main()
