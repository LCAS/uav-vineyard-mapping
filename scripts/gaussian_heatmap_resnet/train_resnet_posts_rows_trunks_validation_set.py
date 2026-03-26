import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np

# --- Model Definition (U-Net with ResNet-18) ---
# No changes are needed in the model architecture itself.
class UNetWithResnet18(nn.Module):
    def __init__(self, n_classes=4): # Default is now 4 for background + 3 classes
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

        # Decoder path with skip connections
        out = F.interpolate(self.upconv1(x6), size=x5.shape[2:], mode='bilinear', align_corners=True)
        out = torch.cat([out, x5], dim=1)
        out = self.conv_up1(out)

        out = F.interpolate(self.upconv2(out), size=x4.shape[2:], mode='bilinear', align_corners=True)
        out = torch.cat([out, x4], dim=1)
        out = self.conv_up2(out)

        out = F.interpolate(self.upconv3(out), size=x3.shape[2:], mode='bilinear', align_corners=True)
        out = torch.cat([out, x3], dim=1)
        out = self.conv_up3(out)

        out = F.interpolate(self.upconv4(out), size=x2.shape[2:], mode='bilinear', align_corners=True)
        out = torch.cat([out, x2], dim=1)
        out = self.conv_up4(out)

        out = self.final_conv(out)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out

# --- Custom Dataset Class (UPDATED) ---
class VineyardDataset(Dataset):
    def __init__(self, image_dirs, mask_dir, image_size=(512, 512), threshold=0.5): # <-- CHANGED: image_dir -> image_dirs
        # self.image_dir = image_dir # No longer needed
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.threshold = threshold
        
        self.image_paths = []
        self.image_filenames = []

        # --- MODIFIED LOOP ---
        # Loop through each folder provided in the list
        for folder in image_dirs:
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(root, file))
                        self.image_filenames.append(file)
        
        # The rest of the __init__ method stays the same
        self.post_mask_map = {}
        self.row_mask_map = {}
        self.trunk_mask_map = {}
        
        for root, _, files in os.walk(mask_dir):
            for file in files:
                original_image_name = None
                if file.startswith("posts_mask_"):
                    original_image_name = file.replace("posts_mask_", "")
                    self.post_mask_map[original_image_name] = os.path.join(root, file)
                elif file.startswith("rows_mask_"):
                    original_image_name = file.replace("rows_mask_", "")
                    self.row_mask_map[original_image_name] = os.path.join(root, file)
                # <-- ADDED: Logic to find vine trunk masks
                elif file.startswith("vine_mask_"):
                    original_image_name = file.replace("vine_mask_", "")
                    self.trunk_mask_map[original_image_name] = os.path.join(root, file)
                    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = self.image_filenames[idx]
        
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.image_size)
        image = np.array(image) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Get paths for all three masks
        post_mask_path = self.post_mask_map.get(img_name)
        row_mask_path = self.row_mask_map.get(img_name)
        trunk_mask_path = self.trunk_mask_map.get(img_name) # <-- ADDED

        # Ensure all masks are found for the given image
        if not all([post_mask_path, row_mask_path, trunk_mask_path]):
            raise FileNotFoundError(f"One or more masks not found for image: {img_name}")

        # Load and process each mask
        post_mask = Image.open(post_mask_path).resize(self.image_size).convert('L')
        row_mask = Image.open(row_mask_path).resize(self.image_size).convert('L')
        trunk_mask = Image.open(trunk_mask_path).resize(self.image_size).convert('L') # <-- ADDED

        post_mask = (np.array(post_mask) / 255.0 > self.threshold).astype(np.uint8)
        row_mask = (np.array(row_mask) / 255.0 > self.threshold).astype(np.uint8)
        trunk_mask = (np.array(trunk_mask) / 255.0 > self.threshold).astype(np.uint8) # <-- ADDED
        
        # --- UPDATED: Combine masks with class IDs ---
        # Class IDs: 0=background, 1=post, 2=row, 3=trunk
        height, width = post_mask.shape
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        
        # The order of assignment determines priority in case of overlap.
        # Here, posts > trunks > rows.
        combined_mask[row_mask == 1] = 2   # Assign rows first
        combined_mask[trunk_mask == 1] = 3 # Trunks overwrite rows
        combined_mask[post_mask == 1] = 1  # Posts overwrite everything
        
        combined_mask = torch.from_numpy(combined_mask).long()

        return image, combined_mask

# --- Main execution block ---
if __name__ == "__main__":
    IMAGE_FOLDERS = [
        "../../images/riseholme/august_2024",
        "../../images/riseholme/march_2025"
    ]
    # Ensure this folder contains masks for posts, rows, AND trunks
    MASK_FOLDER = "./heatmap_masks/riseholme/"

    BATCH_SIZE = 4
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 100
    IMAGE_SIZE = (1280, 960)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # <-- UPDATED: Initialize model with n_classes=4
    # We have 4 classes: 0 (background), 1 (posts), 2 (rows), 3 (trunks)
    model = UNetWithResnet18(n_classes=4).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- Data Splitting ---
    full_dataset = VineyardDataset(image_dirs=IMAGE_FOLDERS, mask_dir=MASK_FOLDER, image_size=IMAGE_SIZE)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_running_loss = 0.0
        for images, masks in train_dataloader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item() * images.size(0)
        
        epoch_train_loss = train_running_loss / len(train_dataloader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, masks in val_dataloader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_running_loss += loss.item() * images.size(0)

        epoch_val_loss = val_running_loss / len(val_dataloader.dataset)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
    
    torch.save(model.state_dict(), "vineyard_detection_model_3class.pth")
    print("Training complete and model saved.")