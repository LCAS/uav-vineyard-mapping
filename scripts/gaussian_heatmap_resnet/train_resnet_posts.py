import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np

# --- Model Definition (U-Net with ResNet-18) ---
class UNetWithResnet18(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        
        # --- Encoder (ResNet-18) ---
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        self.conv1 = self.encoder.conv1
        self.bn1 = self.encoder.bn1
        self.relu = self.encoder.relu
        self.maxpool = self.encoder.maxpool
        self.layer1 = self.encoder.layer1
        self.layer2 = self.encoder.layer2
        self.layer3 = self.encoder.layer3
        self.layer4 = self.encoder.layer4

        # --- Decoder (U-Net style upsampling) ---
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
        # Encoder
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)

        # Decoder
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

# --- Custom Dataset Class ---
class VineyardDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(512, 512), threshold=0.5):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.threshold = threshold
        
        # --- Collect all image paths and filenames, including subfolders ---
        self.image_paths = []
        self.image_filenames = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root, file))
                    self.image_filenames.append(file)
        
        # --- Create a mapping from image name to mask path to handle multiple folders ---
        self.mask_map = {}
        for root, _, files in os.walk(mask_dir):
            for file in files:
                if file.startswith("posts_mask_"):
                    # Extract the original image name from the mask filename
                    original_image_name = file.replace("posts_mask_", "")
                    self.mask_map[original_image_name] = os.path.join(root, file)
                    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Use the stored full path to load the image
        img_path = self.image_paths[idx]
        img_name = self.image_filenames[idx]
        
        # Load the original RGB drone image
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.image_size)
        image = np.array(image) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Load the corresponding mask using the pre-built map
        post_mask_path = self.mask_map.get(img_name)
        if post_mask_path is None:
            raise FileNotFoundError(f"Mask not found for image: {img_name} at {img_path}")

        post_mask = Image.open(post_mask_path).resize(self.image_size).convert('L')
        post_mask = np.array(post_mask) / 255.0
        
        # Convert Gaussian to binary: 0 = background, 1 = post
        post_mask = (post_mask > self.threshold).astype(np.uint8)
        
        # LongTensor for CrossEntropyLoss
        post_mask = torch.from_numpy(post_mask).long()

        return image, post_mask

# --- Main execution block ---
if __name__ == "__main__":
    # This path points to the TOP-LEVEL directory containing all subfolders of drone images
    IMAGE_FOLDER = "../../images/riseholme/"
    # This path point to the TOP-LEVEL directory containing all subfolders of masks
    MASK_FOLDER = "./heatmap_masks/"

    # --- Hyperparameters ---
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    IMAGE_SIZE = (1280, 960)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Instantiate the model and move it to the device
    model = UNetWithResnet18(n_classes=2).to(device)
    
    # Define the loss function (for multi-class pixel classification)
    criterion = nn.CrossEntropyLoss()
    
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Create the dataset and dataloader
    dataset = VineyardDataset(image_dir=IMAGE_FOLDER, mask_dir=MASK_FOLDER, image_size=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # --- Training Loop ---
    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), "vineyard_detection_model.pth")
    print("Training complete and model saved.")