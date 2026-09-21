# -*- coding: UTF-8 -*-
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from WGAN_GP import gradient_penalty, Generator, Discriminator, compute_mmd

# Hyperparameters
noise_dim = 100
hidden_layers_gen = [256, 512]
hidden_layers_disc = [512, 256]
gen_lr = 0.0001
disc_lr = 0.0004
lambda_gp = 10
n_epochs = 1000
critic_iterations = 10

np.random.seed(42)
torch.manual_seed(42)



# Normalize data with MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(data.values)

# Split into training and testing sets while maintaining class proportions
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in sss.split(data_normalized, labels):
    data_train, data_test = data_normalized[train_index], data_normalized[test_index]

data_tensor = torch.tensor(data_train, dtype=torch.float32)
input_dim = data_tensor.shape[1]

generator = Generator(noise_dim, input_dim, hidden_layers_gen)
generator.apply(Generator.weights_init)
discriminator = Discriminator(input_dim, hidden_layers_disc)
discriminator.apply(Discriminator.weights_init)

optimizer_g = optim.Adam(generator.parameters(), lr=gen_lr, betas=(0.5, 0.99))
optimizer_d = optim.Adam(discriminator.parameters(), lr=disc_lr, betas=(0.5, 0.99))

writer = SummaryWriter()

# Tracking losses
g_losses = []
d_losses = []

for epoch in range(n_epochs):
    for _ in range(critic_iterations):
        noise = torch.randn(data_tensor.size(0), noise_dim)
        fake_data = generator(noise).detach()
        real_logits = discriminator(data_tensor)
        fake_logits = discriminator(fake_data)
        gp = gradient_penalty(discriminator, data_tensor, fake_data)
        loss_d = -(real_logits.mean() - fake_logits.mean()) + lambda_gp * gp

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

    noise = torch.randn(data_tensor.size(0), noise_dim)
    fake_data = generator(noise)
    fake_logits = discriminator(fake_data)
    loss_g = -fake_logits.mean()

    optimizer_g.zero_grad()
    loss_g.backward()
    optimizer_g.step()

    # Record losses
    g_losses.append(loss_g.item())
    d_losses.append(loss_d.item())

    # Log losses to TensorBoard
    writer.add_scalar('Loss/Generator', loss_g.item(), epoch)
    writer.add_scalar('Loss/Discriminator', loss_d.item(), epoch)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss D: {loss_d.item()}, Loss G: {loss_g.item()}")

# Plot loss trends
plt.figure(figsize=(10, 5))
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Generator and Discriminator Losses Over Epochs')
plt.legend()
plt.show()

noise = torch.randn(5 * data_tensor.size(0), noise_dim)
fake_data = generator(noise).detach().numpy()

# Convert training data to numpy for visualization
real_data = data_tensor.numpy()

# Combine real and fake data for t-SNE visualization
combined_data = np.vstack([real_data, fake_data])
labels = np.array([0] * real_data.shape[0] + [1] * fake_data.shape[0])  # 0 for real, 1 for fake

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_results = tsne.fit_transform(combined_data)

# Plot t-SNE results
plt.figure(figsize=(10, 8))
plt.scatter(tsne_results[labels == 0, 0], tsne_results[labels == 0, 1], label='Real Data', alpha=0.6, s=30)
plt.scatter(tsne_results[labels == 1, 0], tsne_results[labels == 1, 1], label='Fake Data', alpha=0.6, s=30)
plt.title('t-SNE Visualization of Real and Generated Data')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.show()
