import numpy as np
import torch
import matplotlib.pyplot as plt

from models.neural_network import ChannelEstimator
from data.dataset import generate_dataset


# Configuration
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SNR_LEVELS = [0, 5, 10, 15, 20]

NUM_SAMPLES = 5000
NUM_DATA_BITS = 100

print("Using device:", DEVICE)


# 1. Load trained model
checkpoint = torch.load(
    "models/channel_estimator.pth",
    map_location=DEVICE
)

model = ChannelEstimator().to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

input_mean = checkpoint["input_mean"].to(DEVICE)
input_std = checkpoint["input_std"].to(DEVICE)

print("AI model loaded successfully!")


# Store results
ai_mse = []
ls_mse = []

ai_ber = []
ls_ber = []


# 2. Evaluate at different SNR levels
for snr in SNR_LEVELS:

    print(f"\nEvaluating at SNR = {snr} dB")

    # Generate fresh test dataset
    X, y, ls, _ = generate_dataset(
        num_samples=NUM_SAMPLES,
        num_pilots=16,
        snr_db_range=(snr, snr),
        seed=1000 + snr
    )

    # Prepare AI input
    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE
    )

    X_tensor = (X_tensor - input_mean) / input_std

    # AI prediction
    with torch.no_grad():

        predictions = model(X_tensor)

    h_ai = predictions.cpu().numpy()

    # Actual channel
    h_true = y[:, 0] + 1j * y[:, 1]

    # AI estimated channel
    h_ai_complex = (
        h_ai[:, 0] + 1j * h_ai[:, 1]
    )

    # LS estimated channel
    h_ls_complex = (
        ls[:, 0] + 1j * ls[:, 1]
    )


    # 3. Calculate MSE
    mse_ai = np.mean(
        np.abs(h_true - h_ai_complex) ** 2
    )

    mse_ls = np.mean(
        np.abs(h_true - h_ls_complex) ** 2
    )

    ai_mse.append(mse_ai)
    ls_mse.append(mse_ls)


    # 4. Generate independent BPSK data
    rng = np.random.default_rng(2000 + snr)

    bits = rng.integers(
        0, 2,
        size=(NUM_SAMPLES, NUM_DATA_BITS)
    )

    symbols = 2 * bits - 1


    # 5. Pass data through the same channel
    transmitted = h_true[:, None] * symbols

    snr_linear = 10 ** (snr / 10)

    noise_power = (
        np.abs(h_true[:, None]) ** 2 / snr_linear
    )

    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(transmitted.shape)
        + 1j * rng.standard_normal(transmitted.shape)
    )

    received = transmitted + noise


    # 6. Detect using LS channel estimate
    detected_ls = (
        np.real(received * np.conj(h_ls_complex[:, None])) >= 0
    ).astype(int)

    # Detect using AI channel estimate
    detected_ai = (
        np.real(received * np.conj(h_ai_complex[:, None])) >= 0
    ).astype(int)


    # 7. Calculate BER
    errors_ls = np.sum(detected_ls != bits)
    errors_ai = np.sum(detected_ai != bits)

    total_bits = bits.size

    ber_ls = errors_ls / total_bits
    ber_ai = errors_ai / total_bits

    ls_ber.append(ber_ls)
    ai_ber.append(ber_ai)


    # Print results
    print("LS MSE:", mse_ls)
    print("AI MSE:", mse_ai)

    print("LS BER:", ber_ls)
    print("AI BER:", ber_ai)


# 8. Print final comparison
print("\n========== FINAL RESULTS ==========")

print(
    f"{'SNR':<10}{'LS MSE':<15}"
    f"{'AI MSE':<15}{'LS BER':<15}{'AI BER':<15}"
)

for i, snr in enumerate(SNR_LEVELS):

    print(
        f"{snr:<10}"
        f"{ls_mse[i]:<15.6f}"
        f"{ai_mse[i]:<15.6f}"
        f"{ls_ber[i]:<15.6f}"
        f"{ai_ber[i]:<15.6f}"
    )


# 9. Plot MSE vs SNR
plt.figure(figsize=(8, 5))

plt.plot(
    SNR_LEVELS,
    ls_mse,
    marker="o",
    label="LS Estimation"
)

plt.plot(
    SNR_LEVELS,
    ai_mse,
    marker="s",
    label="AI Estimation"
)

plt.xlabel("SNR (dB)")
plt.ylabel("Mean Squared Error")

plt.title("MSE vs SNR")

plt.yscale("log")
plt.grid(True)
plt.legend()

plt.savefig("mse_comparison.png", dpi=300)

plt.show()


# 10. Plot BER vs SNR
plt.figure(figsize=(8, 5))

plt.semilogy(
    SNR_LEVELS,
    ls_ber,
    marker="o",
    label="LS Detection"
)

plt.semilogy(
    SNR_LEVELS,
    ai_ber,
    marker="s",
    label="AI Detection"
)

plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate")

plt.title("BER vs SNR")

plt.grid(True)
plt.legend()

plt.savefig("ber_comparison.png", dpi=300)

plt.show()


print("\nGraphs saved successfully!")