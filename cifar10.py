import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the discriminator model
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Define the generator model
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Initialize the models and move them to the device
latent_dim = 100
d_model = Discriminator().to(device)
g_model = Generator(latent_dim).to(device)

# Create optimizers
d_optimizer = optim.Adam(d_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = optim.Adam(g_model.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Define the loss function
criterion = nn.BCELoss()

# Define a custom dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.image_files[index])
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image

# Load and prepare image data
dataset_path = "C:/Users/UDAY SANKAR/Desktop/airplane"  # replace with your dataset path
transform = ToTensor()  # define any additional transformations if needed

# Create custom dataset
dataset = CustomDataset(dataset_path, transform=transform)

# Create dataloader
batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
def train(d_model, g_model, dataloader, d_optimizer, g_optimizer, criterion, num_epochs, latent_dim):
    for epoch in range(num_epochs):
        for i, real_images in enumerate(dataloader):
            batch_size = real_images.size(0)
            
            # Move real images to device
            real_images = real_images.to(device)
            
            # Generate fake images
            latent_vectors = torch.randn(batch_size, latent_dim).to(device)
            fake_images = g_model(latent_vectors)
            
            # Train the discriminator
            d_optimizer.zero_grad()
            
            # Real images
            real_labels = torch.ones(batch_size, 1).to(device)
            real_outputs = d_model(real_images)
            d_loss_real = criterion(real_outputs, real_labels)
            
            # Fake images
            fake_labels = torch.zeros(batch_size, 1).to(device)
            fake_outputs = d_model(fake_images.detach())
            d_loss_fake = criterion(fake_outputs, fake_labels)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train the generator
            g_optimizer.zero_grad()
            
            # Generate fake images again
            latent_vectors = torch.randn(batch_size, latent_dim).to(device)
            fake_images = g_model(latent_vectors)
            
            # Re-evaluate the discriminator on the generated images
            fake_outputs = d_model(fake_images)
            
            # Generator loss
            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()
            g_optimizer.step()
            
            # Print training stats
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], "
                      f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")
        
        # Save generated images and models
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                latent_vectors = torch.randn(100, latent_dim).to(device)
                fake_images = g_model(latent_vectors).detach().cpu()
            vutils.save_image(fake_images, f"generated_images_epoch_{epoch+1}.png", normalize=True)
            torch.save(g_model.state_dict(), f"generator_model_epoch_{epoch+1}.pth")

# Train the models
num_epochs = 1000
train(d_model, g_model, dataloader, d_optimizer, g_optimizer, criterion, num_epochs, latent_dim)

# Load the generator model with the best performance
generator_path = f"generator_model_epoch_{num_epochs}.pth"
g_model.load_state_dict(torch.load(generator_path))

# Generate and plot the final images
num_images = 100  # Number of images to generate
with torch.no_grad():
    latent_vectors = torch.randn(num_images, latent_dim).to(device)
    fake_images = g_model(latent_vectors).detach().cpu()

fig = plt.figure(figsize=(10, 10))
for i in range(num_images):
    ax = fig.add_subplot(10, 10, i+1)
    ax.axis("off")
    ax.imshow(fake_images[i].permute(1, 2, 0))
plt.show()

