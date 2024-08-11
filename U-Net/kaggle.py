import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary
import pandas as pd
from torchvision.utils import draw_bounding_boxes


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )


class BloodCellDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_paths = []
        self.df = pd.read_csv(csv_file)
        

        self.image_paths = self.df['image'].unique()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("L")
        

        boxes = self.df[self.df['image'] == img_name][['xmin', 'ymin', 'xmax', 'ymax']].values
        mask = self.create_mask(boxes, image.size)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

    def create_mask(self, boxes, image_size):
        mask = Image.new('L', image_size, 0)
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            for i in range(int(xmin), int(xmax)):
                for j in range(int(ymin), int(ymax)):
                    mask.putpixel((i, j), 255)
        return mask


def train_model(model, dataloader, epochs=5, lr=0.001, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    model.train()

    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

    return losses


def visualize_results(model, dataloader, device='cpu', num_images=5):
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.cpu().squeeze(1).numpy()
            data = data.cpu().squeeze(1).numpy()
            target = target.cpu().squeeze(1).numpy()

            fig, axs = plt.subplots(num_images, 3, figsize=(15, num_images * 5))
            for j in range(num_images):
                axs[j, 0].imshow(data[j], cmap='gray')
                axs[j, 0].set_title('Input Image')
                axs[j, 1].imshow(target[j], cmap='gray')
                axs[j, 1].set_title('Ground Truth')
                axs[j, 2].imshow(output[j], cmap='gray')
                axs[j, 2].set_title('Predicted')

            plt.show()
            break


if __name__ == "__main__":

    model = UNet(in_channels=1, out_channels=1, init_features=32)


    img_dir = r'C:\Users\Lenovo\Desktop\AISC2024 IOTMLLAB\U-Net\archive\images'
    csv_file = r'C:\Users\Lenovo\Desktop\AISC2024 IOTMLLAB\U-Net\archive\annotations.csvq'


    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])


    dataset = BloodCellDataset(img_dir=img_dir, csv_file=csv_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = train_model(model, dataloader, epochs=10, lr=0.001, device=device)

    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


    visualize_results(model, dataloader, device=device)


    summary(model, input_size=(1, 128, 128))

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of parameters: {total_params}')
