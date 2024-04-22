import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import nrrd
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121, resnet18, resnet34, resnet50, SEResNet50, EfficientNetBN
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    Compose,
)
from monai.utils import set_determinism


class CTScansDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(root_dir) if 'signal' in f]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        image, _ = nrrd.read(file_path)
        image = np.max(image, axis=0)  # Maximum intensity projection

        std_path = file_path.replace('signal', 'std')
        std_image, _ = nrrd.read(std_path)
        std_image = np.max(std_image, axis=0)

        image = torch.from_numpy(image).float().unsqueeze(0)  
        std_image = torch.from_numpy(std_image).float().unsqueeze(0)

        if self.transform:
            image = self.transform(image)
            std_image = self.transform(std_image)

        return image, std_image

from monai.transforms import (
    LoadImage, EnsureChannelFirst, ScaleIntensity, RandRotate, RandFlip, RandZoom, Resize,
    Compose, Activations, AsDiscrete
)
from monai.data import ImageDataset, DataLoader

transform = Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Resize((256, 256))
    # LoadImage(image_only=True), 
    # EnsureChannelFirst(),  
    # ScaleIntensity(),  
    # Resize(spatial_size=(256, 256)),  
    # RandRotate(range_x=np.pi/12, prob=0.5, keep_size=True), 
    # RandFlip(spatial_axis=0, prob=0.5), 
    # RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5)
])


data_dir = '/projectnb/ec500kb/projects/Project6/scans'
dataset = CTScansDataset(data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
    
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://discuss.pytorch.org/t/issues-with-torch-nn-functional-pad/19117/4
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)  # Increased number of channels here
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = UNet().to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.MSELoss()
n_channels = 1  
n_classes = 1   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels, n_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


# def train_and_validate(model, train_loader, val_loader, epochs, optimizer, criterion, device, save_path):
#     best_val_loss = float('inf')
#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         for images, std_maps in train_loader:
#             images, std_maps = images.to(device), std_maps.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, std_maps)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

#         avg_train_loss = running_loss / len(train_loader)
#         val_loss = validate(model, val_loader, criterion, device)
#         print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}')

#         # Save the model if the validation loss decreased
#         if val_loss < best_val_loss:
#             print(f'Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f}). Saving model...')
#             torch.save(model.state_dict(), save_path)
#             best_val_loss = val_loss



def train_and_validate(model, train_loader, val_loader, epochs, optimizer, criterion, device, save_path):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, std_maps in train_loader:
            images, std_maps = images.to(device), std_maps.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, std_maps)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        # val_loss = validate(model, val_loader, criterion, device, display_images=True)  
        val_loss = validate(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            print(f'Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f}). Saving model...')
            torch.save(model.state_dict(), save_path)
            best_val_loss = val_loss
        final_images, final_labels, final_outputs = images, std_maps, outputs

    display_sample_images(final_images, final_labels, final_outputs)



import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim




def calculate_psnr_ssim(original, denoised, data_range=255):
    psnr = compare_psnr(original, denoised, data_range=data_range)
    ssim = compare_ssim(original, denoised, data_range=data_range, multichannel=True)
    return psnr, ssim

def validate(model, val_loader, criterion, device, display_images=False):
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Calculate PSNR and SSIM for the current batch
            for j in range(images.size(0)):
                original = images[j].cpu().numpy().squeeze()
                denoised = outputs[j].cpu().numpy().squeeze()
                psnr, ssim = calculate_psnr_ssim(original, denoised)
                total_psnr += psnr
                total_ssim += ssim

            # Optionally display images
            if display_images and i == 0:  # Only display from the first batch
                display_sample_images(images, labels, outputs)

    avg_loss = total_loss / len(val_loader)
    avg_psnr = total_psnr / len(val_loader.dataset)
    avg_ssim = total_ssim / len(val_loader.dataset)
    print(f'Validation Loss: {avg_loss:.4f}, Avg PSNR: {avg_psnr:.2f}, Avg SSIM: {avg_ssim:.4f}')
    return avg_loss

def display_sample_images(images, labels, outputs):
    plt.figure(figsize=(12, 6))
    for i in range(min(3, images.size(0))):  # Display up to 3 images from the batch
        plt.subplot(3, 3, i*3+1)
        plt.imshow(images[i].cpu().detach().numpy().squeeze(), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(3, 3, i*3+2)
        plt.imshow(labels[i].cpu().detach().numpy().squeeze(), cmap='gray')
        plt.title('True STD Map')
        plt.axis('off')

        plt.subplot(3, 3, i*3+3)
        plt.imshow(outputs[i].cpu().detach().numpy().squeeze(), cmap='gray')
        plt.title('Predicted STD Map')
        plt.axis('off')
    plt.savefig('test.png')
    plt.close()

learning_rate = 0.001  
batch_size = 4         
epochs = 30   

model = UNet(n_channels=1, n_classes=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()


save_path = 'out1.test'

train_and_validate(model, train_loader, val_loader, epochs, optimizer, criterion, device, save_path)


# def train(model, train_loader, val_loader, epochs):
#     model.train()
#     for epoch in range(epochs):
#         for images, std_maps in train_loader:
#             images, std_maps = images.to(device), std_maps.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, std_maps)
#             loss.backward()
#             optimizer.step()
#         print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# def test(model, test_loader):
#     model.eval()
#     with torch.no_grad():
#         for images, std_maps in test_loader:
#             images, std_maps = images.to(device), std_maps.to(device)
#             outputs = model(images)

#             plt.figure(figsize=(10, 4))
#             plt.subplot(1, 3, 1)
#             plt.imshow(images[0].cpu().squeeze(), cmap='gray')
#             plt.title('Original Image')
#             plt.subplot(1, 3, 2)
#             plt.imshow(std_maps[0].cpu().squeeze(), cmap='gray')
#             plt.title('True STD Map')
#             plt.subplot(1, 3, 3)
#             plt.imshow(outputs[0].cpu().squeeze(), cmap='gray')
#             plt.title('Predicted STD Map')
#             plt.show()
#             break  


# train(model, train_loader, val_loader, 10)
# test(model, test_loader)
