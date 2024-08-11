import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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


model = UNet(in_channels=1, out_channels=1, init_features=32).to(device)


print(model)

class SimpleDataset(Dataset):
    def __init__(self, size=100, img_size=64):
        self.size = size
        self.img_size = img_size
        self.data = []
        self.targets = []
        for _ in range(size):
            img = np.zeros((img_size, img_size), dtype=np.float32)
            x, y = np.random.randint(0, img_size, size=2)
            radius = np.random.randint(5, 15)
            for i in range(img_size):
                for j in range(img_size):
                    if (i-x)**2 + (j-y)**2 <= radius**2:
                        img[i, j] = 1.0
            self.data.append(img)
            self.targets.append(img)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        return torch.tensor(img).unsqueeze(0), torch.tensor(target).unsqueeze(0)


dataset = SimpleDataset(size=1000, img_size=64)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


def visualize_input_data(dataloader, num_images=5):

    data_iter = iter(dataloader)
    data, _ = next(data_iter)
    

    data = data.cpu().squeeze(1).numpy()
    

    fig, axs = plt.subplots(num_images, 1, figsize=(5, num_images * 5))
    for i in range(num_images):
        axs[i].imshow(data[i], cmap='gray')
        axs[i].set_title(f'Input Image {i + 1}')
        axs[i].axis('off')
    
    plt.show()
    
    
def train_model(model, dataloader, epochs=5, lr=0.001):
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


losses = train_model(model, dataloader, epochs=10, lr=0.001)


plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


def visualize_results(model, dataloader, num_images=5):
    model.eval()
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

visualize_input_data(dataloader, num_images=5)
visualize_results(model, dataloader)


summary(model, input_size=(1, 64, 64))

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total number of parameters: {total_params}')
