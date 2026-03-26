import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torch.amp import autocast, GradScaler

# --------------------------
# Dataset reading from folders
# --------------------------
class VineyardDatasetFromFolders(Dataset):
    def __init__(self, base_dir, split, image_size=(512,512), threshold=0.5):
        self.image_dir = os.path.join(base_dir, split, "images")
        self.row_mask_dir = os.path.join(base_dir, split, "row_masks")
        self.post_mask_dir = os.path.join(base_dir, split, "post_masks")
        self.image_size = image_size
        self.threshold = threshold
        self.file_list = [f for f in os.listdir(self.image_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        img_path = os.path.join(self.image_dir, fname)
        row_mask_path = os.path.join(self.row_mask_dir, fname)
        post_mask_path = os.path.join(self.post_mask_dir, fname)

        image = Image.open(img_path).convert('RGB').resize(self.image_size)
        image = np.array(image)/255.0
        image = torch.from_numpy(image).permute(2,0,1).float()

        row_mask = np.array(Image.open(row_mask_path).resize(self.image_size).convert('L')) / 255.0
        post_mask = np.array(Image.open(post_mask_path).resize(self.image_size).convert('L')) / 255.0
        row_mask = (row_mask>self.threshold).astype(np.uint8)
        post_mask = (post_mask>self.threshold).astype(np.uint8)

        combined_mask = np.zeros_like(post_mask, dtype=np.uint8)
        combined_mask[post_mask==1] = 1
        combined_mask[row_mask==1] = 2
        combined_mask[(post_mask==1) & (row_mask==1)] = 1

        return image, torch.from_numpy(combined_mask).long()

# --------------------------
# U-Net with ResNet18 encoder
# --------------------------
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

# --------------------------
# Training loop with AMP, pinned memory, channels-last
# --------------------------
def train_model(base_dir, batch_size=4, lr=1e-4, epochs=50, image_size=(800,800)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = VineyardDatasetFromFolders(base_dir, "train", image_size=image_size)
    val_ds   = VineyardDatasetFromFolders(base_dir, "val", image_size=image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, pin_memory=True)

    model = UNetResNet18(n_classes=3).to(device, memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler(device="cuda")

    best_val_loss = float('inf')
    best_model_path = "best_vineyard_model.pth"

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                masks = masks.to(device, non_blocking=True)
                with autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with val loss {best_val_loss:.4f}")

    print("Training complete.")

# --------------------------
# Run training
# --------------------------
if __name__ == "__main__":
    BASE_DIR = "dataset_split_augmentation"
    train_model(
        BASE_DIR, 
        batch_size=8, 
        lr=1e-4, 
        epochs=200, 
        image_size=(800,800)
    )
