"""Verify generated Massive MIMO dataset dimensions, pilot orthogonality, and baseline LS/MMSE NMSE."""

import argparse
import numpy as np

from config.config import DEFAULT_PATHS
from data.dataset import load_mimo_dataset
from traditional.mmse_estimator import mimo_mmse_estimate
from utils.pilots import verify_pilot_orthogonality


def inspect_dataset(dataset_path: str) -> None:
    data = load_mimo_dataset(dataset_path)
    H = data["H"]
    X = data["X"]
    Y = data["Y"]
    snr_db = data["snr_db"]
    noise_var = data["noise_variance"]
    h_ls = data["h_ls"]
    inputs = data["inputs"]
    targets = data["targets"]

    nt = int(data["nt"])
    nr = int(data["nr"])
    tp = int(data["tp"])

    print(f"Dataset path       : {dataset_path}")
    print(f"Antenna config     : Nt={nt}, Nr={nr}, Tp={tp}")
    print(f"Total samples      : {H.shape[0]}")
    print(f"H shape / dtype    : {H.shape} / {H.dtype}")
    print(f"X shape / dtype    : {X.shape} / {X.dtype}")
    print(f"Y shape / dtype    : {Y.shape} / {Y.dtype}")
    print(f"inputs shape       : {inputs.shape} (expected {(H.shape[0], 2 * nr * tp)})")
    print(f"targets shape      : {targets.shape} (expected {(H.shape[0], 2 * nr * nt)})")

    assert inputs.shape == (H.shape[0], 2 * nr * tp)
    assert targets.shape == (H.shape[0], 2 * nr * nt)

    is_ortho, ortho_err = verify_pilot_orthogonality(X)
    print(f"Pilot orthogonal   : {is_ortho} (max |X@X^H - Tp*I| = {ortho_err:.3e})")

    # Check empirical channel & noise statistics
    avg_h_power = float(np.mean(np.abs(H) ** 2))
    print(f"Mean E[|H_ij|^2]   : {avg_h_power:.4f} (theoretical Rayleigh = 1.0000)")

    h_mmse = mimo_mmse_estimate(Y, X, noise_var)

    nmse_ls = float(np.mean(np.abs(H - h_ls) ** 2) / np.mean(np.abs(H) ** 2))
    nmse_mmse = float(np.mean(np.abs(H - h_mmse) ** 2) / np.mean(np.abs(H) ** 2))
    print(f"Average LS NMSE    : {nmse_ls:.6f} ({10.0 * np.log10(nmse_ls):.2f} dB)")
    print(f"Average MMSE NMSE  : {nmse_mmse:.6f} ({10.0 * np.log10(nmse_mmse):.2f} dB)")
    print(f"SNR range (dB)     : [{snr_db.min():.2f}, {snr_db.max():.2f}]")
    print("Dataset verification successful!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Massive MIMO dataset archive.")
    parser.add_argument(
        "--path",
        type=str,
        default=str(DEFAULT_PATHS.default_dataset_path),
        help="Path to NPZ dataset file.",
    )
    args = parser.parse_args()
    inspect_dataset(args.path)
