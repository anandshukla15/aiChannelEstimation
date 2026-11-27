"""Massive MIMO dataset generator and complex-to-real flattening utilities.

Flattening convention (row-major / C-order):
    Given complex batch A of shape (B, Nr, K):
        A_real_flat = A.real.reshape(B, Nr * K)
        A_imag_flat = A.imag.reshape(B, Nr * K)
        features    = np.concatenate([A_real_flat, A_imag_flat], axis=1)  # (B, 2 * Nr * K)

    To reconstruct complex tensor A of shape (B, Nr, K):
        half = Nr * K
        A_complex = features[:, :half].reshape(B, Nr, K) + 1j * features[:, half:].reshape(B, Nr, K)
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np

from config.config import DEFAULT_MIMO, DEFAULT_PATHS, DEFAULT_XL_MIMO
from traditional.ls_estimator import mimo_ls_estimate
from utils.channel import mimo_rayleigh_channel, xl_mimo_channel_batch
from utils.noise import compute_noise_variance, generate_complex_awgn
from utils.pilots import generate_pilots


def complex_to_real_features(tensor: np.ndarray) -> np.ndarray:
    """Flatten complex array of shape (B, Nr, K) into float32 (B, 2 * Nr * K).

    Order: [Re(row-major flatten), Im(row-major flatten)].
    """
    arr = np.asarray(tensor)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D tensor (B, Nr, K), got shape {arr.shape}")
    batch_size = arr.shape[0]
    real_part = arr.real.reshape(batch_size, -1)
    imag_part = arr.imag.reshape(batch_size, -1)
    return np.concatenate([real_part, imag_part], axis=1).astype(np.float32)


def real_features_to_complex(
    features: np.ndarray,
    nr: int,
    cols: int,
) -> np.ndarray:
    """Reconstruct complex array (B, Nr, cols) from real features (B, 2 * Nr * cols)."""
    arr = np.asarray(features)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D features (B, 2*Nr*cols), got shape {arr.shape}")
    expected_dim = 2 * nr * cols
    if arr.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch: expected {expected_dim} (2*{nr}*{cols}), got {arr.shape[1]}"
        )
    batch_size = arr.shape[0]
    half = nr * cols
    real_part = arr[:, :half].reshape(batch_size, nr, cols)
    imag_part = arr[:, half:].reshape(batch_size, nr, cols)
    return real_part + 1j * imag_part


def generate_mimo_dataset(
    num_samples: int = DEFAULT_MIMO.num_train_val_samples,
    nt: int = DEFAULT_MIMO.nt,
    nr: int = DEFAULT_MIMO.nr,
    tp: int = DEFAULT_MIMO.tp,
    snr_db_range: Tuple[float, float] = DEFAULT_MIMO.snr_db_range,
    pilot_type: str = DEFAULT_MIMO.pilot_type,
    require_orthogonal: bool = True,
    channel_model: str = "rayleigh",
    xl_regime: str = "near_field",
    seed: int = DEFAULT_MIMO.train_seed,
    compute_ls: bool = True,
) -> Dict[str, Union[np.ndarray, int, str]]:
    """Generate a Massive MIMO (or XL-MIMO) dataset for Y = H @ X + N.

    Returns a dictionary containing:
        H: (num_samples, nr, nt) complex64 actual channels
        X: (nt, tp) complex64 pilot matrix
        Y: (num_samples, nr, tp) complex64 received signals
        snr_db: (num_samples,) float32 SNR in dB per sample
        noise_variance: (num_samples,) float32 sigma^2 per sample
        h_ls: (num_samples, nr, nt) complex64 LS estimates (if compute_ls=True)
        inputs: (num_samples, 2 * nr * tp) float32 flattened [Re(Y), Im(Y)]
        targets: (num_samples, 2 * nr * nt) float32 flattened [Re(H), Im(H)]
        nt, nr, tp, seed, pilot_type, channel_model metadata
    """
    rng = np.random.default_rng(seed)

    # 1. Generate pilot matrix X of shape (nt, tp)
    pilots = generate_pilots(
        nt=nt,
        tp=tp,
        method=pilot_type,
        normalized=False,
        require_orthogonal=require_orthogonal,
    )

    # 2. Generate complex channel matrices H of shape (num_samples, nr, nt)
    if channel_model == "rayleigh":
        h_true = mimo_rayleigh_channel(num_samples=num_samples, nr=nr, nt=nt, rng=rng, dtype=np.complex128)
    elif channel_model == "xl_mimo":
        dist_range = (
            DEFAULT_XL_MIMO.near_field_distance_range_m
            if xl_regime == "near_field"
            else (DEFAULT_XL_MIMO.far_field_distance_m, DEFAULT_XL_MIMO.far_field_distance_m)
        )
        h_true, _ = xl_mimo_channel_batch(
            num_samples=num_samples,
            nr=nr,
            nt=nt,
            wavelength_m=DEFAULT_XL_MIMO.wavelength_m,
            spacing_m=DEFAULT_XL_MIMO.antenna_spacing_m,
            distance_range_m=dist_range,
            angle_range_deg=DEFAULT_XL_MIMO.angle_range_deg,
            non_stationary_visibility=(
                DEFAULT_XL_MIMO.non_stationary_visibility if xl_regime == "near_field" else False
            ),
            visibility_fraction_range=DEFAULT_XL_MIMO.visibility_fraction_range,
            rician_k_factor_db=DEFAULT_XL_MIMO.rician_k_factor_db,
            regime=xl_regime,
            rng=rng,
        )
    else:
        raise ValueError(f"Unsupported channel_model='{channel_model}'. Use 'rayleigh' or 'xl_mimo'.")

    # 3. Pilot transmission: Y_clean = H @ X -> (num_samples, nr, tp)
    y_clean = h_true @ pilots

    # 4. SNR and noise variance: E[|(HX)_{i,k}|^2] = tr(X @ X^H) / tp (= Nt for unnormalized pilots)
    expected_sig_power = float(np.real(np.trace(pilots @ pilots.conj().T)) / tp)
    snr_db = rng.uniform(snr_db_range[0], snr_db_range[1], size=num_samples)
    noise_variance = compute_noise_variance(snr_db, signal_power=expected_sig_power)

    # 5. Complex AWGN N = sqrt(sigma^2 / 2) * (N_r + j N_i)
    noise = generate_complex_awgn(
        shape=(num_samples, nr, tp),
        noise_variance=noise_variance[:, None, None],
        rng=rng,
        dtype=np.complex128,
    )

    y_received = y_clean + noise

    # 6. LS channel estimate
    h_ls = mimo_ls_estimate(y_received, pilots) if compute_ls else None

    # 7. Real-valued NN inputs and targets
    inputs = complex_to_real_features(y_received)
    targets = complex_to_real_features(h_true)

    dataset: Dict[str, Union[np.ndarray, int, str]] = {
        "H": h_true.astype(np.complex64),
        "X": pilots.astype(np.complex64),
        "Y": y_received.astype(np.complex64),
        "snr_db": snr_db.astype(np.float32),
        "noise_variance": np.asarray(noise_variance, dtype=np.float32),
        "inputs": inputs,
        "targets": targets,
        "nt": np.int32(nt),
        "nr": np.int32(nr),
        "tp": np.int32(tp),
        "seed": np.int32(seed),
        "pilot_type": np.array(pilot_type),
        "channel_model": np.array(channel_model if channel_model == "rayleigh" else f"xl_mimo_{xl_regime}"),
    }
    if h_ls is not None:
        dataset["h_ls"] = h_ls.astype(np.complex64)

    return dataset


def save_mimo_dataset(
    output_path: Union[str, Path],
    dataset: Dict[str, Union[np.ndarray, int, str]],
) -> Path:
    """Save generated MIMO dataset to compressed NPZ format."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)
    return path


def load_mimo_dataset(dataset_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load MIMO dataset from NPZ archive."""
    with np.load(dataset_path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def generate_dataset(
    num_samples: int = DEFAULT_MIMO.num_train_val_samples,
    num_pilots: int = DEFAULT_MIMO.tp,
    snr_db_range: Tuple[float, float] = DEFAULT_MIMO.snr_db_range,
    seed: int = DEFAULT_MIMO.train_seed,
    nt: int = DEFAULT_MIMO.nt,
    nr: int = DEFAULT_MIMO.nr,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility helper returning (inputs, targets, ls_flat, snr_db) for MIMO."""
    ds = generate_mimo_dataset(
        num_samples=num_samples,
        nt=nt,
        nr=nr,
        tp=num_pilots,
        snr_db_range=snr_db_range,
        seed=seed,
        compute_ls=True,
    )
    ls_flat = complex_to_real_features(ds["h_ls"])
    return ds["inputs"], ds["targets"], ls_flat, ds["snr_db"]


if __name__ == "__main__":
    DEFAULT_PATHS.ensure_dirs()
    print(
        f"Generating Massive MIMO train/val dataset: "
        f"Nt={DEFAULT_MIMO.nt}, Nr={DEFAULT_MIMO.nr}, Tp={DEFAULT_MIMO.tp}, "
        f"N={DEFAULT_MIMO.num_train_val_samples}, SNR={DEFAULT_MIMO.snr_db_range} dB"
    )
    train_val_ds = generate_mimo_dataset(
        num_samples=DEFAULT_MIMO.num_train_val_samples,
        nt=DEFAULT_MIMO.nt,
        nr=DEFAULT_MIMO.nr,
        tp=DEFAULT_MIMO.tp,
        snr_db_range=DEFAULT_MIMO.snr_db_range,
        pilot_type=DEFAULT_MIMO.pilot_type,
        seed=DEFAULT_MIMO.train_seed,
    )
    save_mimo_dataset(DEFAULT_PATHS.default_dataset_path, train_val_ds)

    print(
        f"Generating independent Massive MIMO test dataset: "
        f"N={DEFAULT_MIMO.num_test_samples}, seed={DEFAULT_MIMO.test_seed}"
    )
    test_ds = generate_mimo_dataset(
        num_samples=DEFAULT_MIMO.num_test_samples,
        nt=DEFAULT_MIMO.nt,
        nr=DEFAULT_MIMO.nr,
        tp=DEFAULT_MIMO.tp,
        snr_db_range=DEFAULT_MIMO.snr_db_range,
        pilot_type=DEFAULT_MIMO.pilot_type,
        seed=DEFAULT_MIMO.test_seed,
    )
    save_mimo_dataset(DEFAULT_PATHS.default_test_dataset_path, test_ds)

    print("Saved train/val dataset:", DEFAULT_PATHS.default_dataset_path)
    print("Saved test dataset:", DEFAULT_PATHS.default_test_dataset_path)
    print("H shape:", train_val_ds["H"].shape)
    print("X shape:", train_val_ds["X"].shape)
    print("Y shape:", train_val_ds["Y"].shape)
    print("NN inputs shape:", train_val_ds["inputs"].shape)
    print("NN targets shape:", train_val_ds["targets"].shape)
