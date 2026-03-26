import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import numpy as np
import os
import cv2
from tqdm import tqdm
import shutil
import yaml

# ======================
# CONFIGURATION
# ======================
MODEL_PATH = "./results_resnet/yolo_masks/train_resnet18_20251031_153442/resnet18_unet_image_size_1014x760_batch_size_8.pth"
INPUT_DIR = "../../images/riseholme/july_2025/39_feet/"
OUTPUT_DIR = "inference_resnet_yolo_labels/heatmaps/riseholme/trained_on_vineyard_segmentation_20/inference_image_size_4056x3040/best_model/july_2025/39_feet/"

# IMAGE_SIZE = (1280, 960) # (width, height) full res images are 3040x4056
IMAGE_SIZE = (4056, 3040)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Class configuration is now loaded from the YAML file ---
# NUM_CLASSES = 4
# CLASS_NAMES = ["background", "pole", "trunk", "vine_row"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# LOAD TRAINING CONFIG
# ======================
CONFIG_PATH = os.path.join(os.path.dirname(MODEL_PATH), "training_config.yaml")
try:
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(f"✅ Loaded training config from: {CONFIG_PATH}")

    # Extract parameters from the loaded config
    NUM_CLASSES = config['num_classes']
    CLASS_NAMES = config['class_names']
    BACKBONE = config['backbone']
    TRAIN_IMAGE_SIZE = tuple(config['image_size'])
    
    print(f"   -> Backbone: {BACKBONE}")
    print(f"   -> Num Classes: {NUM_CLASSES}")
    print(f"   -> Class Names: {CLASS_NAMES}")

    # Warn if inference image size doesn't match training
    if TRAIN_IMAGE_SIZE != IMAGE_SIZE:
        print(f"⚠️ WARNING: Inference IMAGE_SIZE {IMAGE_SIZE} does not match")
        print(f"   the training size {TRAIN_IMAGE_SIZE} from config.")
        print(f"   Resizing to {IMAGE_SIZE} but this may affect performance.")

    # Copy the config to the output directory for traceability
    DEST_CONFIG_PATH = os.path.join(OUTPUT_DIR, "training_config_used_for_inference.yaml")
    shutil.copy(CONFIG_PATH, DEST_CONFIG_PATH)
    print(f"   -> Copied config for this run to: {DEST_CONFIG_PATH}")

except FileNotFoundError:
    print(f"❌ ERROR: Could not find 'training_config.yaml' at '{CONFIG_PATH}'")
    print("   Cannot proceed without config. Please ensure it's in the same directory as the model.")
    exit()
except Exception as e:
    print(f"❌ ERROR reading config file: {e}")
    exit()

# ======================
# MODEL DEFINITION (Flexible, from training script) <-- UPDATED SECTION
# ======================
class UNetResNet(nn.Module):
    def __init__(self, n_classes, backbone="resnet18"):
        super().__init__()

        # Choose backbone
        if backbone == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            enc_ch = [64, 64, 128, 256, 512]  # channels after each stage
        elif backbone == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            enc_ch = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Backbone '{backbone}' not supported")

        # Encoder
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool0 = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder: in_channels = upconv_out_channels + skip_channels
        self.upconv4 = nn.ConvTranspose2d(enc_ch[4], enc_ch[3], 2, stride=2)
        self.dec4 = nn.Conv2d(enc_ch[3] + enc_ch[3], enc_ch[3], 3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(enc_ch[3], enc_ch[2], 2, stride=2)
        self.dec3 = nn.Conv2d(enc_ch[2] + enc_ch[2], enc_ch[2], 3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(enc_ch[2], enc_ch[1], 2, stride=2)
        self.dec2 = nn.Conv2d(enc_ch[1] + enc_ch[1], enc_ch[1], 3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(enc_ch[1], enc_ch[0], 2, stride=2)
        self.dec1 = nn.Conv2d(enc_ch[0] + enc_ch[0], 64, 3, padding=1)  # final decoder before output

        # Final classification layer
        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        # Encoder
        x0 = self.encoder0(x)
        x1 = self.pool0(x0)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        # Decoder
        d4 = F.interpolate(self.upconv4(x5), size=x4.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.dec4(d4)

        d3 = F.interpolate(self.upconv3(d4), size=x3.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)

        d2 = F.interpolate(self.upconv2(d3), size=x2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        d1 = F.interpolate(self.upconv1(d2), size=x0.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out

# ======================
# LOAD MODEL
# ======================
# Now, we initialize the model using the parameters from the config
model = UNetResNet(n_classes=NUM_CLASSES, backbone=BACKBONE).to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Error: Model file not found at {MODEL_PATH}")
    exit()
except RuntimeError as e:
    print(f"❌ Error loading model state_dict. This can happen if the model architecture in the script")
    print(f"   does not match the one that was saved. Check that NUM_CLASSES and BACKBONE are correct.")
    print(f"   Original error: {e}")
    exit()

# ======================
# VISUALIZATION FUNCTION
# ======================
def generate_single_class_heatmap(image_path, output_path, model_outputs, class_idx, threshold=0.5, alpha=0.6):
    """
    Generates and saves a heatmap overlay for a single specified class.
    """
    image_pil = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    image_np = np.array(image_pil)
    
    class_prob_map = model_outputs[class_idx, :, :]
    
    mask = class_prob_map >= threshold
    if not np.any(mask):
        return

    normalized_prob = np.zeros_like(class_prob_map)
    detected_probs = class_prob_map[mask]
    if detected_probs.max() > detected_probs.min():
        normalized_prob[mask] = (detected_probs - detected_probs.min()) / (detected_probs.max() - detected_probs.min())
    else:
        normalized_prob[mask] = 1.0
        
    heatmap_gray = (normalized_prob * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    output_image = image_np.copy()
    blended_part = cv2.addWeighted(image_np, 1 - alpha, heatmap_color, alpha, 0)
    output_image[mask] = blended_part[mask]

    Image.fromarray(output_image).save(output_path)

# ======================
# RUN DETECTION
# ======================
print(f"\nStarting inference process on images from '{INPUT_DIR}'...")
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for file in tqdm(image_files, desc="Processing images"):
    img_path = os.path.join(INPUT_DIR, file)
    
    image_pil = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
    image_np_for_tensor = np.array(image_pil) / 255.0
    image_tensor = torch.from_numpy(image_np_for_tensor).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    # Use the CLASS_NAMES loaded from the config
    for idx, name in enumerate(CLASS_NAMES[1:], start=1):
        file_basename = os.path.splitext(file)[0]
        out_path = os.path.join(OUTPUT_DIR, f"{file_basename}_heatmap_{name}.jpg")
        
        generate_single_class_heatmap(img_path, out_path, probs, idx, threshold=0.3, alpha=0.6)

print(f"\n✅ All separate heatmaps generated successfully in '{OUTPUT_DIR}'!")