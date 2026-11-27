"""Experiment 2: Effect of Transmit Antenna Count (Nt in {8, 16, 32, 64})."""

from typing import Dict, List, Optional
import numpy as np
import torch

from config.config import DEFAULT_EXPERIMENT, DEFAULT_MIMO, DEFAULT_PATHS
from data.dataset import generate_mimo_dataset
from evaluation.metrics import compute_nmse, count_parameters, measure_inference_time
from evaluation.visualization import plot_metric_vs_x, save_csv
from traditional.ls_estimator import mimo_ls_estimate
from traditional.mmse_estimator import mimo_mmse_estimate
from train import predict_mimo_channel, train_channel_estimator


def run_antenna_experiment(
    nt_list: Optional[List[int]] = None,
    nr: int = DEFAULT_EXPERIMENT.fixed_nr,
    eval_snr_db: float = 10.0,
    train_samples: int = DEFAULT_EXPERIMENT.sweep_train_samples,
    test_samples: int = DEFAULT_EXPERIMENT.sweep_test_samples,
    epochs: int = DEFAULT_EXPERIMENT.sweep_epochs,
) -> List[Dict[str, object]]:
    """Compare LS, MMSE, and AI-DNN across Nt = 8, 16, 32, 64 with Tp = Nt."""
    DEFAULT_PATHS.ensure_dirs()
    if nt_list is None:
        nt_list = list(DEFAULT_EXPERIMENT.antenna_nt_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: List[Dict[str, object]] = []
    ls_nmse_db_list, mmse_nmse_db_list, ai_nmse_db_list = [], [], []
    ls_time_list, mmse_time_list, ai_time_list = [], [], []

    print(f"\n=== EXPERIMENT 2: Antenna Configuration Sweep (Nr={nr}, Eval SNR={eval_snr_db} dB) ===")
    for nt in nt_list:
        tp = nt
        train_ds = generate_mimo_dataset(
            num_samples=train_samples,
            nt=nt,
            nr=nr,
            tp=tp,
            snr_db_range=DEFAULT_MIMO.snr_db_range,
            seed=4100 + nt,
            compute_ls=False,
        )
        ckpt_path = DEFAULT_PATHS.checkpoints_dir / f"exp_ant_nt{nt}_nr{nr}_tp{tp}.pth"
        train_res = train_channel_estimator(
            inputs=train_ds["inputs"],
            targets=train_ds["targets"],
            nr=nr,
            nt=nt,
            tp=tp,
            checkpoint_path=ckpt_path,
            epochs=epochs,
            seed=42 + nt,
            backup_existing=False,
            verbose=False,
        )
        model = train_res["model"].to(device)
        input_mean = torch.from_numpy(train_res["input_mean"]).to(device)
        input_std = torch.from_numpy(train_res["input_std"]).to(device)
        num_params = count_parameters(model)

        test_ds = generate_mimo_dataset(
            num_samples=test_samples,
            nt=nt,
            nr=nr,
            tp=tp,
            snr_db_range=(eval_snr_db, eval_snr_db),
            seed=5100 + nt,
            compute_ls=False,
        )
        h_true = test_ds["H"].astype(np.complex128)
        pilots = test_ds["X"].astype(np.complex128)
        y_rx = test_ds["Y"].astype(np.complex128)
        noise_var = test_ds["noise_variance"].astype(np.float64)
        inputs = test_ds["inputs"]

        t_ls_ms = measure_inference_time(lambda: mimo_ls_estimate(y_rx, pilots), num_runs=10, warmup_runs=2)
        h_ls = mimo_ls_estimate(y_rx, pilots)

        t_mmse_ms = measure_inference_time(
            lambda: mimo_mmse_estimate(y_rx, pilots, noise_var), num_runs=10, warmup_runs=2
        )
        h_mmse = mimo_mmse_estimate(y_rx, pilots, noise_var)

        t_ai_ms = measure_inference_time(
            lambda: predict_mimo_channel(model, inputs, input_mean, input_std, nr=nr, nt=nt, device=device),
            num_runs=10,
            warmup_runs=2,
        )
        h_ai = predict_mimo_channel(model, inputs, input_mean, input_std, nr=nr, nt=nt, device=device)

        nmse_ls, nmse_ls_db = compute_nmse(h_true, h_ls)
        nmse_mmse, nmse_mmse_db = compute_nmse(h_true, h_mmse)
        nmse_ai, nmse_ai_db = compute_nmse(h_true, h_ai)

        ls_nmse_db_list.append(nmse_ls_db)
        mmse_nmse_db_list.append(nmse_mmse_db)
        ai_nmse_db_list.append(nmse_ai_db)
        ls_time_list.append(t_ls_ms)
        mmse_time_list.append(t_mmse_ms)
        ai_time_list.append(t_ai_ms)

        rows.append(
            {
                "nt": nt,
                "nr": nr,
                "tp": tp,
                "eval_snr_db": eval_snr_db,
                "ls_nmse_db": nmse_ls_db,
                "mmse_nmse_db": nmse_mmse_db,
                "ai_dnn_nmse_db": nmse_ai_db,
                "ls_nmse_lin": nmse_ls,
                "mmse_nmse_lin": nmse_mmse,
                "ai_dnn_nmse_lin": nmse_ai,
                "ls_time_ms": t_ls_ms,
                "mmse_time_ms": t_mmse_ms,
                "ai_dnn_time_ms": t_ai_ms,
                "ai_dnn_params": num_params,
            }
        )
        print(
            f"Nt={nt:2d}, Tp={tp:2d} | Params={num_params:7,d} | "
            f"NMSE(dB): LS={nmse_ls_db:6.2f}, MMSE={nmse_mmse_db:6.2f}, AI={nmse_ai_db:6.2f} | "
            f"Time(ms): LS={t_ls_ms:.3f}, MMSE={t_mmse_ms:.3f}, AI={t_ai_ms:.3f}"
        )

    save_csv(DEFAULT_PATHS.tables_dir / "antenna_experiment.csv", rows)
    plot_metric_vs_x(
        nt_list,
        {"LS": ls_nmse_db_list, "MMSE": mmse_nmse_db_list, "AI-DNN": ai_nmse_db_list},
        xlabel="Number of Transmit Antennas (Nt = Tp)",
        ylabel="NMSE (dB)",
        title=f"NMSE vs Transmit Antenna Count (Nr={nr}, SNR={eval_snr_db:.0f} dB)",
        output_path=DEFAULT_PATHS.plots_dir / "nmse_vs_antennas.png",
        logy=False,
    )
    plot_metric_vs_x(
        nt_list,
        {"LS": ls_time_list, "MMSE": mmse_time_list, "AI-DNN": ai_time_list},
        xlabel="Number of Transmit Antennas (Nt = Tp)",
        ylabel="Batch Inference Time (ms)",
        title=f"Inference Time vs Transmit Antenna Count ({test_samples} samples, Nr={nr})",
        output_path=DEFAULT_PATHS.plots_dir / "inference_time_vs_antennas.png",
        logy=False,
    )
    return rows


if __name__ == "__main__":
    run_antenna_experiment()
