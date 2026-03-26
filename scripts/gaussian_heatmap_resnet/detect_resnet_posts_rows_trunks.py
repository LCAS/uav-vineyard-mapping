import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np
import cv2

# --- Model Definition (U-Net with ResNet-18 encoder) ---
# This class remains unchanged.
class UNetWithResnet18(nn.Module):
    def __init__(self, n_classes=4):
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
        
        return out

# --- Helper Function for Heatmap Generation ---
def create_blended_heatmap(prob_map, original_pil_image, output_path):
    """Generates a colored heatmap and blends it with the original image."""
    original_size = original_pil_image.size # (width, height)
    heatmap_normalized = (prob_map * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap_normalized, original_size, interpolation=cv2.INTER_LINEAR)
    
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)
    heatmap_pil = Image.fromarray(heatmap_colored_rgb)

    blended_img = Image.blend(original_pil_image, heatmap_pil, alpha=0.6)
    blended_img.save(output_path)

# --- Main Inference Script (MODIFIED for Batch Processing) ---
if __name__ == "__main__":
    MODEL_PATH = "./models/riseholme/august_2024_march_2025/all_altitudes/posts_rows_trunks_background/vineyard_detection_model_posts_rows_background_image_size_1280x960_batch_size_4_epochs_100.pth"
    
    IMAGE_FOLDER = "../../images/riseholme/july_2025/100_feet/"

    OUTPUT_FOLDER = "./inference_outputs_heatmaps/riseholme/july_2025/100_feet/"
    IMAGE_SIZE = (1280, 960)
    NUM_CLASSES = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model only once
    model = UNetWithResnet18(n_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.eval()
    print("Model loaded and set to evaluation mode.")
    
    # --- ADDED: Loop through all images in the folder ---
    for image_name in os.listdir(IMAGE_FOLDER):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue # Skip non-image files

        print(f"\nProcessing image: {image_name}...")
        image_path = os.path.join(IMAGE_FOLDER, image_name)
        
        # Create a unique output subfolder for each image to avoid filename conflicts
        image_output_folder = os.path.join(OUTPUT_FOLDER, os.path.splitext(image_name)[0])
        os.makedirs(image_output_folder, exist_ok=True)

        # --- Core inference logic is now inside the loop ---
        image = Image.open(image_path).convert('RGB')
        original_img_np = np.array(image)
        original_pil = Image.fromarray(original_img_np)
        
        image_resized_for_model = image.resize(IMAGE_SIZE)
        image_np_for_model = np.array(image_resized_for_model)
        image_tensor = torch.from_numpy(image_np_for_model / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)

        # Extract Probability Maps for Each Class
        post_probs = probs[0, 1, :, :].cpu().numpy()
        row_probs = probs[0, 2, :, :].cpu().numpy()
        trunk_probs = probs[0, 3, :, :].cpu().numpy()

        # Generate and Save Blended Heatmaps
        create_blended_heatmap(post_probs, original_pil, os.path.join(image_output_folder, "heatmap_blended_posts.png"))
        create_blended_heatmap(row_probs, original_pil, os.path.join(image_output_folder, "heatmap_blended_rows.png"))
        create_blended_heatmap(trunk_probs, original_pil, os.path.join(image_output_folder, "heatmap_blended_trunks.png"))

        # Generate the Combined Semantic Segmentation Mask
        pred_classes = torch.argmax(probs, dim=1)[0].cpu().numpy().astype(np.uint8)
        pred_classes_resized = cv2.resize(pred_classes, original_pil.size, interpolation=cv2.INTER_NEAREST)

        colored_mask = np.zeros_like(original_img_np)
        colored_mask[pred_classes_resized == 1] = [255, 0, 0]  # Red for posts
        colored_mask[pred_classes_resized == 2] = [0, 255, 0]  # Green for rows
        colored_mask[pred_classes_resized == 3] = [0, 0, 255]  # Blue for trunks
        Image.fromarray(colored_mask).save(os.path.join(image_output_folder, "semantic_mask_colored.png"))

        # Also save a blended version of the semantic mask
        semantic_blended_img = Image.blend(original_pil, Image.fromarray(colored_mask), alpha=0.5)
        semantic_blended_img.save(os.path.join(image_output_folder, "semantic_mask_blended.png"))

        print(f"Outputs for {image_name} saved to {image_output_folder}")
        
    print("\nAll images processed.")