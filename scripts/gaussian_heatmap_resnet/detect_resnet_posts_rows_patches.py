import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np
import cv2

# --- Model Definition (Three-class U-Net with ResNet-18 encoder) ---
class UNetResNet18(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up4 = nn.Conv2d(512, 256, 3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up3 = nn.Conv2d(256, 128, 3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up2 = nn.Conv2d(128, 64, 3, padding=1)
        self.upconv1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv_up1 = nn.Conv2d(128, 64, 3, padding=1)
        self.final_conv = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)

        up4 = F.interpolate(self.upconv4(x6), size=x5.shape[2:], mode='bilinear', align_corners=True)
        up4 = self.conv_up4(torch.cat([up4, x5], dim=1))
        up3 = F.interpolate(self.upconv3(up4), size=x4.shape[2:], mode='bilinear', align_corners=True)
        up3 = self.conv_up3(torch.cat([up3, x4], dim=1))
        up2 = F.interpolate(self.upconv2(up3), size=x3.shape[2:], mode='bilinear', align_corners=True)
        up2 = self.conv_up2(torch.cat([up2, x3], dim=1))
        up1 = F.interpolate(self.upconv1(up2), size=x2.shape[2:], mode='bilinear', align_corners=True)
        up1 = self.conv_up1(torch.cat([up1, x2], dim=1))
        out = self.final_conv(F.interpolate(up1, size=x.shape[2:], mode='bilinear', align_corners=True))
        return out

# --- Main Inference Script (background + posts + rows) ---
if __name__ == "__main__":
    # MODEL_PATH = "./models/patch_images/vineyard_detection_model_posts_rows_background_image_size_800x800_batch_size_4_epochs_23_50.pth"
    # MODEL_PATH = "./models/patch_images/vineyard_detection_model_posts_rows_background_image_size_800x800_batch_size_4_epochs_104_200.pth"
    MODEL_PATH = "./models/patch_images/vineyard_detection_model_posts_rows_background_image_size_800x800_batch_size_8_epochs_200_200.pth"
    # NEW_IMAGE_PATH = "dataset/orthophoto/jojo/may_2025/patches/3_6.png"
    # NEW_IMAGE_PATH = "dataset/orthophoto/arun_valley/may_2025/patches/11_6.png"
    # NEW_IMAGE_PATH = "dataset/orthophoto/arun_valley/may_2025/patches/16_4.png"
    # NEW_IMAGE_PATH = "dataset/orthophoto/riseholme/august_2024/39_feet/patches/2_1.png"
    # NEW_IMAGE_PATH = "dataset/orthophoto/riseholme/august_2024/100_feet/patches/1_0.png"
    # NEW_IMAGE_PATH = "dataset/orthophoto/coolhurst/may_2025_bad/patches/7_2.png"
    NEW_IMAGE_PATH = "dataset/orthophoto/gusbourne/may_2025/patches/5_9.png"
    OUTPUT_FOLDER = "./inference_outputs/"
    IMAGE_SIZE = (800, 800)
    NUM_CLASSES = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load model
    model = UNetResNet18(n_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # Preprocess image
    image = Image.open(NEW_IMAGE_PATH).convert('RGB')
    original_img_np = np.array(image.resize(IMAGE_SIZE))
    image_tensor = torch.from_numpy(original_img_np / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        outputs = model(image_tensor)  # (1, 3, H, W)
        probs = torch.softmax(outputs, dim=1)

    # Predicted class map
    pred_classes = torch.argmax(probs, dim=1)[0].cpu().numpy()
    pred_classes_resized = cv2.resize(pred_classes.astype(np.uint8), IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

    print(f"Prediction mask shape: {pred_classes_resized.shape}")
    print(f"Unique values in mask: {np.unique(pred_classes_resized)}")

    # --- Otsu thresholding for posts and rows ---
    post_probs = probs[0, 1, :, :].cpu().numpy()
    row_probs  = probs[0, 2, :, :].cpu().numpy()

    post_heatmap_normalized = (post_probs * 255).astype(np.uint8)
    row_heatmap_normalized  = (row_probs * 255).astype(np.uint8)

    _, otsu_post = cv2.threshold(post_heatmap_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_row  = cv2.threshold(row_heatmap_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    otsu_post_resized = cv2.resize(otsu_post, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    otsu_row_resized  = cv2.resize(otsu_row, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

    Image.fromarray(otsu_post_resized).save(os.path.join(OUTPUT_FOLDER, "binary_post_mask_otsu.png"))
    Image.fromarray(otsu_row_resized).save(os.path.join(OUTPUT_FOLDER, "binary_row_mask_otsu.png"))

    # --- Colored semantic map ---
    colored_mask = np.zeros((*pred_classes_resized.shape, 3), dtype=np.uint8)
    colored_mask[pred_classes_resized == 1] = [255, 0, 0]  # posts = red
    colored_mask[pred_classes_resized == 2] = [0, 255, 0]  # rows = green
    Image.fromarray(colored_mask).save(os.path.join(OUTPUT_FOLDER, "semantic_mask_colored.png"))

    # --- Blended overlay ---
    original_img_pil = Image.fromarray(original_img_np).convert("RGB")
    colored_mask_pil = Image.fromarray(colored_mask).convert("RGB")
    blended_img = Image.blend(original_img_pil, colored_mask_pil, alpha=0.5)
    blended_img.save(os.path.join(OUTPUT_FOLDER, "blended_detection_output_test.png"))

    print(f"Inference complete. Outputs saved to {OUTPUT_FOLDER}")
