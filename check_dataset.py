import numpy as np

data = np.load("data/channel_dataset.npz")

X = data["X"]
y = data["y"]
ls = data["ls"]
snr_db = data["snr_db"]

print("Total samples:", len(X))
print("Input features:", X.shape[1])
print("Output features:", y.shape[1])

print("\nAverage LS MSE:")

mse = np.mean(np.sum((y - ls) ** 2, axis=1))

print(mse)

print("\nSNR range:")
print("Minimum:", snr_db.min())
print("Maximum:", snr_db.max())

print("\nDataset verification successful!")