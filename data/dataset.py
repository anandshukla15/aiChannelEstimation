import numpy as np


def generate_dataset(
    num_samples=50000,
    num_pilots=16,
    snr_db_range=(0, 20),
    seed=42
):

    rng = np.random.default_rng(seed)

    # Fixed known BPSK pilot sequence
    pilots = np.ones((num_samples, num_pilots))

    # Generate Rayleigh fading channel
    real = rng.standard_normal((num_samples, 1))
    imag = rng.standard_normal((num_samples, 1))

    h = (real + 1j * imag) / np.sqrt(2)

    # Transmit pilots
    transmitted = h * pilots

    # Generate random SNR
    snr_db = rng.uniform(
        snr_db_range[0],
        snr_db_range[1],
        size=(num_samples, 1)
    )

    snr_linear = 10 ** (snr_db / 10)

    # Calculate noise power
    noise_power = 1.0 / snr_linear

    # Generate complex AWGN
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal((num_samples, num_pilots))
        + 1j * rng.standard_normal((num_samples, num_pilots))
    )

    # Received signal
    received = transmitted + noise

    # Neural network input
    X = np.concatenate(
        [received.real, received.imag],
        axis=1
    )

    # Actual channel
    y = np.concatenate(
        [h.real, h.imag],
        axis=1
    )

    # LS estimation
    numerator = np.sum(
        np.conj(pilots) * received,
        axis=1
    )

    denominator = np.sum(
        np.abs(pilots) ** 2,
        axis=1
    )

    h_ls = numerator / denominator

    ls = np.column_stack([
        h_ls.real,
        h_ls.imag
    ])

    return X, y, ls, snr_db.flatten()


if __name__ == "__main__":

    print("Generating corrected dataset...")

    X, y, ls, snr_db = generate_dataset()

    np.savez_compressed(
        "data/channel_dataset.npz",
        X=X,
        y=y,
        ls=ls,
        snr_db=snr_db
    )

    print("Corrected dataset saved!")

    print("Input:", X.shape)
    print("Target:", y.shape)
    print("LS:", ls.shape)