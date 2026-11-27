"""End-to-end Massive MIMO & 6G XL-MIMO simulation entry point."""

import argparse
import numpy as np

from config.config import DEFAULT_MIMO, DEFAULT_PATHS, DEFAULT_XL_MIMO
from data.dataset import generate_mimo_dataset, save_mimo_dataset
from evaluation.metrics import compute_mse, compute_nmse, simulate_mimo_ber
from traditional.ls_estimator import mimo_ls_estimate
from traditional.mmse_estimator import mimo_mmse_estimate
from train import load_trained_estimator, predict_mimo_channel, train_channel_estimator
from utils.channel import xl_mimo_channel_batch
from utils.pilots import verify_pilot_orthogonality


def run_demo() -> None:
    """Quick physical sanity demonstration of Massive MIMO and 6G XL-MIMO channels."""
    DEFAULT_PATHS.ensure_dirs()
    snr_db = 15.0
    demo_ds = generate_mimo_dataset(
        num_samples=500,
        nt=DEFAULT_MIMO.nt,
        nr=DEFAULT_MIMO.nr,
        tp=DEFAULT_MIMO.tp,
        snr_db_range=(snr_db, snr_db),
        seed=999,
    )
    H = demo_ds["H"].astype(np.complex128)
    X = demo_ds["X"].astype(np.complex128)
    Y = demo_ds["Y"].astype(np.complex128)
    noise_var = demo_ds["noise_variance"].astype(np.float64)

    is_ortho, ortho_err = verify_pilot_orthogonality(X)
    h_ls = mimo_ls_estimate(Y, X)
    h_mmse = mimo_mmse_estimate(Y, X, noise_var)

    _, nmse_ls_db = compute_nmse(H, h_ls)
    _, nmse_mmse_db = compute_nmse(H, h_mmse)
    ber_ls = simulate_mimo_ber(H, h_ls, snr_db=snr_db, num_data_symbols=30, seed=1)
    ber_mmse = simulate_mimo_ber(H, h_mmse, snr_db=snr_db, num_data_symbols=30, seed=1)

    print("=== Massive MIMO Demo (Nt=32, Nr=4, Tp=32, SNR=15 dB) ===")
    print(f"Pilot orthogonality X@X^H = Tp*I : {is_ortho} (err={ortho_err:.2e})")
    print(f"LS   NMSE: {nmse_ls_db:6.2f} dB | BER: {ber_ls:.5f}")
    print(f"MMSE NMSE: {nmse_mmse_db:6.2f} dB | BER: {ber_mmse:.5f}")

    ckpt = DEFAULT_PATHS.checkpoints_dir / f"mimo_dnn_nt{DEFAULT_MIMO.nt}_nr{DEFAULT_MIMO.nr}_tp{DEFAULT_MIMO.tp}.pth"
    if ckpt.exists():
        model, mean, std, _ = load_trained_estimator(ckpt)
        h_ai = predict_mimo_channel(model, demo_ds["inputs"], mean, std, nr=DEFAULT_MIMO.nr, nt=DEFAULT_MIMO.nt)
        _, nmse_ai_db = compute_nmse(H, h_ai)
        ber_ai = simulate_mimo_ber(H, h_ai, snr_db=snr_db, num_data_symbols=30, seed=1)
        print(f"AI   NMSE: {nmse_ai_db:6.2f} dB | BER: {ber_ai:.5f}")
    else:
        print("AI checkpoint not found yet; run `python train.py` first.")

    # Quick XL-MIMO near-field check
    h_nf, meta = xl_mimo_channel_batch(
        num_samples=50,
        nr=DEFAULT_XL_MIMO.nr,
        nt=DEFAULT_XL_MIMO.nt,
        wavelength_m=DEFAULT_XL_MIMO.wavelength_m,
        spacing_m=DEFAULT_XL_MIMO.antenna_spacing_m,
        distance_range_m=DEFAULT_XL_MIMO.near_field_distance_range_m,
        regime="near_field",
        rng=np.random.default_rng(7),
    )
    print(
        f"\n=== 6G XL-MIMO Near-Field Check (Nt={DEFAULT_XL_MIMO.nt}, f_c=28 GHz, "
        f"Rayleigh dist={DEFAULT_XL_MIMO.rayleigh_distance_m:.2f} m) ==="
    )
    print(f"XL-MIMO H shape: {h_nf.shape}, mean distance: {meta['distances_m'].mean():.2f} m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="6G-Oriented Massive MIMO Channel Estimation")
    parser.add_argument("--run-pipeline", action="store_true", help="Generate data, train DNN, and run all experiments.")
    args = parser.parse_args()

    if args.run_pipeline:
        DEFAULT_PATHS.ensure_dirs()
        ds = generate_mimo_dataset()
        save_mimo_dataset(DEFAULT_PATHS.default_dataset_path, ds)
        test_ds = generate_mimo_dataset(
            num_samples=DEFAULT_MIMO.num_test_samples,
            seed=DEFAULT_MIMO.test_seed,
        )
        save_mimo_dataset(DEFAULT_PATHS.default_test_dataset_path, test_ds)
        ckpt = DEFAULT_PATHS.checkpoints_dir / f"mimo_dnn_nt{DEFAULT_MIMO.nt}_nr{DEFAULT_MIMO.nr}_tp{DEFAULT_MIMO.tp}.pth"
        train_channel_estimator(
            inputs=ds["inputs"],
            targets=ds["targets"],
            nr=DEFAULT_MIMO.nr,
            nt=DEFAULT_MIMO.nt,
            tp=DEFAULT_MIMO.tp,
            checkpoint_path=ckpt,
        )
        from evaluate import main as eval_main
        import sys
        sys.argv = ["evaluate.py", "--experiment", "all", "--checkpoint", str(ckpt)]
        eval_main()
    else:
        run_demo()
