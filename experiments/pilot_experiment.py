"""Experiment 4: Pilot Overhead Analysis (varying pilot length Tp for Nt=32, Nr=4).

Mathematical note on pilot orthogonality:
    Row orthogonality X @ X^H = Tp * I_{Nt} requires rank(X) = Nt, which is
    mathematically IMPOSSIBLE whenever Tp < Nt because rank(X) <= min(Nt, Tp) = Tp < Nt.
    For Tp < Nt (underdetermined regime, e.g., Tp in {8, 16} when Nt=32):
      - We use full-column-rank Hadamard-column BPSK pilots (X^H @ X = Nt * I_{Tp}).
      - Exact unbiased LS (X @ X^H)^{-1} does not exist because X @ X^H is singular;
        we evaluate minimum-norm LS (Y @ pinv(X)), which suffers a fundamental
        null-space projection error floor even at high SNR.
      - Regularized MMSE Y @ X^H @ (X @ X^H + sigma^2 I)^{-1} remains well-defined.
"""

from typing import Dict, List, Optional
import numpy as np
import torch

from config.config import DEFAULT_EXPERIMENT, DEFAULT_MIMO, DEFAULT_PATHS
from data.dataset import generate_mimo_dataset
from evaluation.metrics import compute_nmse
from evaluation.visualization import plot_metric_vs_x, save_csv
from traditional.ls_estimator import mimo_ls_estimate
from traditional.mmse_estimator import mimo_mmse_estimate
from train import predict_mimo_channel, train_channel_estimator


def run_pilot_experiment(
    nt: int = DEFAULT_MIMO.nt,
    nr: int = DEFAULT_MIMO.nr,
    pilot_lengths: Optional[List[int]] = None,
    eval_snr_db: float = 15.0,
    train_samples: int = DEFAULT_EXPERIMENT.sweep_train_samples,
    test_samples: int = DEFAULT_EXPERIMENT.sweep_test_samples,
    epochs: int = DEFAULT_EXPERIMENT.sweep_epochs,
) -> List[Dict[str, object]]:
    """Evaluate LS, MMSE, and AI-DNN across pilot lengths Tp = 8, 16, 32, 64."""
    DEFAULT_PATHS.ensure_dirs()
    if pilot_lengths is None:
        pilot_lengths = list(DEFAULT_EXPERIMENT.pilot_lengths_for_nt32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: List[Dict[str, object]] = []
    ls_nmse_db_list, mmse_nmse_db_list, ai_nmse_db_list = [], [], []

    print(f"\n=== EXPERIMENT 4: Pilot Overhead Sweep (Nt={nt}, Nr={nr}, Eval SNR={eval_snr_db} dB) ===")
    for tp in pilot_lengths:
        is_ortho_possible = tp >= nt
        train_ds = generate_mimo_dataset(
            num_samples=train_samples,
            nt=nt,
            nr=nr,
            tp=tp,
            snr_db_range=DEFAULT_MIMO.snr_db_range,
            require_orthogonal=is_ortho_possible,
            seed=6100 + tp,
            compute_ls=False,
        )
        ckpt_path = DEFAULT_PATHS.checkpoints_dir / f"exp_pilot_nt{nt}_nr{nr}_tp{tp}.pth"
        train_res = train_channel_estimator(
            inputs=train_ds["inputs"],
            targets=train_ds["targets"],
            nr=nr,
            nt=nt,
            tp=tp,
            checkpoint_path=ckpt_path,
            epochs=epochs,
            seed=42 + tp,
            backup_existing=False,
            verbose=False,
        )
        model = train_res["model"].to(device)
        input_mean = torch.from_numpy(train_res["input_mean"]).to(device)
        input_std = torch.from_numpy(train_res["input_std"]).to(device)

        test_ds = generate_mimo_dataset(
            num_samples=test_samples,
            nt=nt,
            nr=nr,
            tp=tp,
            snr_db_range=(eval_snr_db, eval_snr_db),
            require_orthogonal=is_ortho_possible,
            seed=7100 + tp,
            compute_ls=False,
        )
        h_true = test_ds["H"].astype(np.complex128)
        pilots = test_ds["X"].astype(np.complex128)
        y_rx = test_ds["Y"].astype(np.complex128)
        noise_var = test_ds["noise_variance"].astype(np.float64)

        h_ls = mimo_ls_estimate(y_rx, pilots)
        h_mmse = mimo_mmse_estimate(y_rx, pilots, noise_var)
        h_ai = predict_mimo_channel(model, test_ds["inputs"], input_mean, input_std, nr=nr, nt=nt, device=device)

        nmse_ls, nmse_ls_db = compute_nmse(h_true, h_ls)
        nmse_mmse, nmse_mmse_db = compute_nmse(h_true, h_mmse)
        nmse_ai, nmse_ai_db = compute_nmse(h_true, h_ai)

        ls_nmse_db_list.append(nmse_ls_db)
        mmse_nmse_db_list.append(nmse_mmse_db)
        ai_nmse_db_list.append(nmse_ai_db)

        rows.append(
            {
                "tp": tp,
                "nt": nt,
                "nr": nr,
                "orthogonal_possible": is_ortho_possible,
                "eval_snr_db": eval_snr_db,
                "ls_nmse_db": nmse_ls_db,
                "mmse_nmse_db": nmse_mmse_db,
                "ai_dnn_nmse_db": nmse_ai_db,
                "ls_nmse_lin": nmse_ls,
                "mmse_nmse_lin": nmse_mmse,
                "ai_dnn_nmse_lin": nmse_ai,
            }
        )
        regime_note = "Orthogonal" if is_ortho_possible else "Underdetermined (pinv LS)"
        print(
            f"Tp={tp:2d} ({regime_note:25s}) | "
            f"NMSE(dB): LS={nmse_ls_db:6.2f}, MMSE={nmse_mmse_db:6.2f}, AI={nmse_ai_db:6.2f}"
        )

    save_csv(DEFAULT_PATHS.tables_dir / "pilot_experiment.csv", rows)
    plot_metric_vs_x(
        pilot_lengths,
        {"LS (pinv if Tp<Nt)": ls_nmse_db_list, "MMSE": mmse_nmse_db_list, "AI-DNN": ai_nmse_db_list},
        xlabel="Pilot Length Tp (symbols)",
        ylabel="NMSE (dB)",
        title=f"NMSE vs Pilot Overhead Tp (Nt={nt}, Nr={nr}, SNR={eval_snr_db:.0f} dB)",
        output_path=DEFAULT_PATHS.plots_dir / "nmse_vs_pilots.png",
        logy=False,
    )
    return rows


if __name__ == "__main__":
    run_pilot_experiment()
