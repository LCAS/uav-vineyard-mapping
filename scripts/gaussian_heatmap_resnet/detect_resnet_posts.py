import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np
import cv2

# --- Model Definition (Two-class U-Net with ResNet-18 encoder) ---
class UNetWithResnet18(nn.Module):
    def __init__(self, n_classes=2):  # Two classes: background, post
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

# --- Main Inference Script (background + post) ---
if __name__ == "__main__":
    MODEL_PATH = "./models/patch_images/vineyard_detection_model_posts_rows_background_image_size_800x800_batch_size_4_epochs_50.pth"
    # NEW_IMAGE_PATH = "../../images/riseholme/august_2024/39_feet/DJI_20240802142923_0025_W.JPG"
    # NEW_IMAGE_PATH = "../../images/riseholme/march_2025/39_feet/DJI_20250310145027_0078_W.JPG"
    NEW_IMAGE_PATH = "../../images/riseholme/august_2024/65_feet/DJI_20240802140739_0043_W.JPG"
    OUTPUT_FOLDER = "./inference_outputs/"
    IMAGE_SIZE = (4056, 3040)  # keep original resolution

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load model
    model = UNetWithResnet18(n_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # Load and preprocess image
    image = Image.open(NEW_IMAGE_PATH).convert('RGB')
    image = image.resize(IMAGE_SIZE)
    original_img_np = np.array(image)
    image_tensor = torch.from_numpy(original_img_np / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        outputs = model(image_tensor)  # shape: (1, 2, H, W)
        probs = torch.softmax(outputs, dim=1)  # convert logits to probabilities

    # Get predicted class map
    pred_classes = torch.argmax(probs, dim=1)[0].cpu().numpy()  # 0=background, 1=post

    # Get post probability heatmap (channel 1)
    post_probs = probs[0, 1, :, :].cpu().numpy()
    post_heatmap_normalized = (post_probs * 255).astype(np.uint8)

    CUSTOM_THRESHOLD = 0.1
    # Apply the threshold to the probability map to create the binary mask
    binary_post_mask = (post_probs > CUSTOM_THRESHOLD).astype(np.uint8) * 255
    Image.fromarray(binary_post_mask).save(os.path.join(OUTPUT_FOLDER, f"binary_post_mask_threshold_{CUSTOM_THRESHOLD:.2f}.png"))

    # --- Apply Otsu's method for automatic thresholding ---
    # The `cv2.threshold` function will return two values:
    # 1. The optimal threshold value found by the algorithm.
    # 2. The resulting binary mask.
    optimal_threshold, otsu_binary_mask = cv2.threshold(
        post_heatmap_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    print(f"Otsu's optimal threshold value: {optimal_threshold}")

    # Save the binary mask created by Otsu's method
    Image.fromarray(otsu_binary_mask).save(os.path.join(OUTPUT_FOLDER,  f"binary_post_mask_threshold_otsi_{optimal_threshold:.2f}.png"))

    # Save heatmap
    post_heatmap_image = Image.fromarray(post_heatmap_normalized)
    post_heatmap_image.save(os.path.join(OUTPUT_FOLDER, "post_probability_heatmap.png"))

    # Create and save colored heatmap
    heatmap_colored_np_post = cv2.cvtColor(
        cv2.applyColorMap(post_heatmap_normalized, cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB
    )
    Image.fromarray(heatmap_colored_np_post).save(os.path.join(OUTPUT_FOLDER, "post_heatmap_colored.png"))

    # Blend heatmap with original image
    original_img_pil = Image.fromarray(original_img_np).convert("RGB")
    post_heatmap_colored_pil = Image.fromarray(heatmap_colored_np_post).resize(original_img_pil.size)
    blended_post_img = Image.blend(original_img_pil, post_heatmap_colored_pil, alpha=0.5)
    blended_post_img.save(os.path.join(OUTPUT_FOLDER, "blended_post_output.png"))

    print(f"Inference complete. Outputs saved to {OUTPUT_FOLDER}")
