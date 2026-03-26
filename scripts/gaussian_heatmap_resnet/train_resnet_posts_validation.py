import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as T
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau # Import the scheduler

# --- Model Definition (U-Net with ResNet-18) ---
class UNetWithResnet18(nn.Module):
    def __init__(self, n_classes=2):
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

# --- Custom Dataset Class (with deterministic augmentation) ---
class VineyardDataset(Dataset):
    def __init__(self, all_image_paths, all_mask_map, image_size=(512, 512), threshold=0.5):
        self.image_size = image_size
        self.threshold = threshold
        
        self.post_mask_map = all_mask_map
        
        self.image_paths = []
        self.image_filenames = []
        self.transforms = []

        for original_path in all_image_paths:
            original_name = os.path.basename(original_path)
            for rotation in [0, 90, 180, 270]:
                self.image_paths.append(original_path)
                self.image_filenames.append(original_name)
                self.transforms.append({'rotation': rotation, 'flip': False})
                
                self.image_paths.append(original_path)
                self.image_filenames.append(original_name)
                self.transforms.append({'rotation': rotation, 'flip': True})
                    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = self.image_filenames[idx]
        current_transform = self.transforms[idx]
        
        image = Image.open(img_path).convert('RGB')
        post_mask = Image.open(self.post_mask_map.get(img_name)).convert('L')
        
        image = image.resize(self.image_size)
        post_mask = post_mask.resize(self.image_size)

        if current_transform['rotation'] > 0:
            image = image.rotate(current_transform['rotation'])
            post_mask = post_mask.rotate(current_transform['rotation'])
        
        if current_transform['flip']:
            image = T.functional.hflip(image)
            post_mask = T.functional.hflip(post_mask)

        image = np.array(image) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        post_mask = np.array(post_mask) / 255.0
        post_mask = (post_mask > self.threshold).astype(np.uint8)
        post_mask = torch.from_numpy(post_mask).long()

        return image, post_mask

# --- Main execution block ---
if __name__ == "__main__":
    # IMAGE_FOLDER = "../../images/riseholme/august_2024/"
    # IMAGE_FOLDER = "../../images/riseholme/march_2025/"
    # IMAGE_FOLDER = "../../images/riseholme/"

    # 1. Define all image and mask folder paths
    IMAGE_FOLDERS = [
        "../../images/riseholme/",
        "../../images/agri_tech_centre/jojo/"
    ]
    MASK_FOLDER = "./heatmap_masks/"

    BATCH_SIZE = 8
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 100
    IMAGE_SIZE = (1280, 960)

    # PATIENCE is now part of the scheduler
    # The early stopping logic will be based on a new counter
    PATIENCE = 60

    # 2. Consolidate image and mask paths from all specified folders
    all_image_paths = []
    for image_folder in IMAGE_FOLDERS:
        for root, _, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_image_paths.append(os.path.join(root, file))

    all_mask_map = {}
    for root, _, files in os.walk(MASK_FOLDER):
        for file in files:
            if file.startswith("posts_mask_"):
                original_image_name = file.replace("posts_mask_", "")
                all_mask_map[original_image_name] = os.path.join(root, file)

    print(f"Found {len(all_image_paths)} images across all folders.")
    
    # 3. Create the dataset with the consolidated lists
    full_dataset = VineyardDataset(
        all_image_paths=all_image_paths, 
        all_mask_map=all_mask_map, 
        image_size=IMAGE_SIZE
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = UNetWithResnet18(n_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- Initialize the scheduler ---
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    best_model_path = "best_vineyard_detection_model.pth"
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        train_running_loss = 0.0
        for images, masks in train_dataloader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
        
        # --- Update the learning rate scheduler with the validation loss ---
        scheduler.step(epoch_val_loss)
        
        end_time = time.time()
        epoch_duration = end_time - start_time
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f} (Time: {epoch_duration:.2f}s)")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Saving new best model with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break
    
    print("Training complete.")
    print(f"Best model saved to {best_model_path}")