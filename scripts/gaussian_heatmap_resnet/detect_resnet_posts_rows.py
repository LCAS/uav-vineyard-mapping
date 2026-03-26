import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np
import cv2

# --- Model Definition (Three-class U-Net with ResNet-18 encoder) ---
class UNetWithResnet18(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        self.conv1 = self.encoder.conv1
        self.bn1 = self.encoder.bn1
        self.relu = self.encoder.relu
        self.maxpool = self.encoder.maxpool
        self.layer1 = self.encoder.layer1
        self.layer2 = self.encoder.layer2
        self.layer3 = self.encoder.layer3
        self.layer4 = self.encoder.layer4

        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv_up4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)

        out = F.interpolate(self.upconv1(x6), size=x5.shape[2:], mode='bilinear', align_corners=True)
        out = torch.cat([out, x5], dim=1)
        out = F.relu(self.conv_up1(out))

        out = F.interpolate(self.upconv2(out), size=x4.shape[2:], mode='bilinear', align_corners=True)
        out = torch.cat([out, x4], dim=1)
        out = F.relu(self.conv_up2(out))

        out = F.interpolate(self.upconv3(out), size=x3.shape[2:], mode='bilinear', align_corners=True)
        out = torch.cat([out, x3], dim=1)
        out = F.relu(self.conv_up3(out))

        out = F.interpolate(self.upconv4(out), size=x2.shape[2:], mode='bilinear', align_corners=True)
        out = torch.cat([out, x2], dim=1)
        out = F.relu(self.conv_up4(out))

        out = self.final_conv(out)
        
        return out  # raw logits

# --- Main Inference Script (background + post + rows) ---
if __name__ == "__main__":
    MODEL_PATH = "./models/riseholme/august_2024_march_2024_all_altitudes/posts_rows_background/vineyard_detection_model_posts_rows_background_image_size_1280x960_batch_size_4_epochs_100.pth"
    NEW_IMAGE_PATH = "../../images/riseholme/august_2024/39_feet/DJI_20240802142923_0025_W.JPG"
    # NEW_IMAGE_PATH = "../../images/riseholme/march_2025/39_feet/DJI_20250310145027_0078_W.JPG"
    OUTPUT_FOLDER = "./inference_outputs/"
    IMAGE_SIZE = (4056, 3040)
    NUM_CLASSES = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    model = UNetWithResnet18(n_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    image = Image.open(NEW_IMAGE_PATH).convert('RGB')
    original_img_np = np.array(image.resize(IMAGE_SIZE))
    image_tensor = torch.from_numpy(original_img_np / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)

    pred_classes = torch.argmax(probs, dim=1)[0].cpu().numpy()

    print(f"Shape of pred_classes: {pred_classes.shape}")
    print(f"Unique values in pred_classes before resizing: {np.unique(pred_classes)}")

    # --- Ensure correct resizing of the prediction mask ---
    # The interpolation method cv2.INTER_NEAREST is used for masks to preserve class labels.
    pred_classes_resized = cv2.resize(pred_classes.astype(np.uint8), IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)    

    print(f"Shape of pred_classes_resized: {pred_classes_resized.shape}")
    print(f"Unique values in pred_classes_resized after resizing: {np.unique(pred_classes_resized)}")

    # --- Apply Otsu's method to each class's probability map individually ---
    post_probs = probs[0, 1, :, :].cpu().numpy()
    row_probs = probs[0, 2, :, :].cpu().numpy()

    post_heatmap_normalized = (post_probs * 255).astype(np.uint8)
    row_heatmap_normalized = (row_probs * 255).astype(np.uint8)

    optimal_threshold_post, otsu_binary_mask_post = cv2.threshold(
        post_heatmap_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"Otsu's optimal threshold for posts: {optimal_threshold_post}")
    otsu_binary_mask_post_resized = cv2.resize(otsu_binary_mask_post, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    Image.fromarray(otsu_binary_mask_post_resized).save(
        os.path.join(OUTPUT_FOLDER, f"binary_post_mask_otsu_{optimal_threshold_post:.2f}.png")
    )

    optimal_threshold_row, otsu_binary_mask_row = cv2.threshold(
        row_heatmap_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"Otsu's optimal threshold for rows: {optimal_threshold_row}")
    otsu_binary_mask_row_resized = cv2.resize(otsu_binary_mask_row, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    Image.fromarray(otsu_binary_mask_row_resized).save(
        os.path.join(OUTPUT_FOLDER, f"binary_row_mask_otsu_{optimal_threshold_row:.2f}.png")
    )

    # --- Create a single colored segmentation map for all classes ---
    # Use the shape of the resized prediction mask to create the colored mask
    colored_mask = np.zeros(
        (pred_classes_resized.shape[0], pred_classes_resized.shape[1], 3),
        dtype=np.uint8
    )
    # The boolean indexing here is correct as long as the dimensions match
    colored_mask[pred_classes_resized == 1] = [255, 0, 0]  # Red for posts
    colored_mask[pred_classes_resized == 2] = [0, 255, 0]  # Green for rows
    Image.fromarray(colored_mask).save(os.path.join(OUTPUT_FOLDER, "semantic_mask_colored.png"))

    # --- Blend the colored segmentation mask with the original image ---
    original_img_pil = Image.fromarray(original_img_np).convert("RGB")
    colored_mask_pil = Image.fromarray(colored_mask).convert("RGB")
    blended_img = Image.blend(original_img_pil, colored_mask_pil, alpha=0.5)
    blended_img.save(os.path.join(OUTPUT_FOLDER, "blended_detection_output.png"))

    print(f"Inference complete. Outputs saved to {OUTPUT_FOLDER}")