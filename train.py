import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from models.neural_network import ChannelEstimator


# Configuration
BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# 1. Load dataset
data = np.load("data/channel_dataset.npz")

X = data["X"].astype(np.float32)
y = data["y"].astype(np.float32)


# 2. Convert into tensors
X = torch.tensor(X)
y = torch.tensor(y)


# 3. Split dataset: 80% training, 20% validation
dataset = TensorDataset(X, y)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)


# 4. Normalize inputs using training data only
train_indices = train_dataset.indices

X_train = X[train_indices]

input_mean = X_train.mean(dim=0)
input_std = X_train.std(dim=0).clamp_min(1e-6)

# Apply same normalization to both datasets
X = (X - input_mean) / input_std

dataset = TensorDataset(X, y)

train_dataset = torch.utils.data.Subset(
    dataset, train_indices
)

val_dataset = torch.utils.data.Subset(
    dataset, val_dataset.indices
)


# 5. Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# 6. Initialize model
model = ChannelEstimator().to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# 7. Training loop
for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0

    for inputs, targets in train_loader:

        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(inputs)

        loss = criterion(predictions, targets)

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_dataset)


    # 8. Validation
    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for inputs, targets in val_loader:

            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            predictions = model(inputs)

            loss = criterion(predictions, targets)

            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_dataset)


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.6f} "
        f"Validation Loss: {val_loss:.6f}"
    )


# 9. Save trained model and normalization parameters
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_mean": input_mean,
        "input_std": input_std
    },
    "models/channel_estimator.pth"
)

print("\nTraining completed!")
print("Model saved at models/channel_estimator.pth")