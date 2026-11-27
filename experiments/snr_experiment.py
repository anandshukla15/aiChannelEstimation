"""Experiment 1: NMSE, MSE, BER, and Inference Time vs SNR (0 to 30 dB)."""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch

from config.config import DEFAULT_EXPERIMENT, DEFAULT_MIMO, DEFAULT_PATHS
from data.dataset import generate_mimo_dataset
from evaluation.metrics import (
    compute_mse,
    compute_nmse,
    count_parameters,
    measure_inference_time,
    simulate_mimo_ber,
)
from evaluation.visualization import plot_inference_time_bar, plot_metric_vs_x, save_csv
from traditional.ls_estimator import mimo_ls_estimate
from traditional.mmse_estimator import mimo_mmse_estimate
from train import load_trained_estimator, predict_mimo_channel


def run_snr_experiment(
    checkpoint_path: Optional[Path] = None,
    snr_levels: Optional[List[float]] = None,
    num_test_samples: int = DEFAULT_MIMO.num_test_samples,
    save_outputs: bool = True,
) -> List[Dict[str, object]]:
    """Evaluate LS, MMSE, and AI-DNN across SNR levels on independent test sets."""
    DEFAULT_PATHS.ensure_dirs()
    if snr_levels is None:
        snr_levels = list(DEFAULT_EXPERIMENT.snr_levels)
    if checkpoint_path is None:
        checkpoint_path = (
            DEFAULT_PATHS.checkpoints_dir
            / f"mimo_dnn_nt{DEFAULT_MIMO.nt}_nr{DEFAULT_MIMO.nr}_tp{DEFAULT_MIMO.tp}.pth"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, input_mean, input_std, ckpt = load_trained_estimator(checkpoint_path, device=device)
    nt = int(ckpt["mimo_config"]["nt"])
    nr = int(ckpt["mimo_config"]["nr"])
    tp = int(ckpt["mimo_config"]["tp"])
    num_params = count_parameters(model)

    rows: List[Dict[str, object]] = []
    ls_nmse_db_list, mmse_nmse_db_list, ai_nmse_db_list = [], [], []
    ls_mse_list, mmse_mse_list, ai_mse_list = [], [], []
    ls_ber_list, mmse_ber_list, ai_ber_list, perfect_ber_list = [], [], [], []
    ls_time_list, mmse_time_list, ai_time_list = [], [], []

    print(f"\n=== EXPERIMENT 1: NMSE / MSE / BER vs SNR (Nt={nt}, Nr={nr}, Tp={tp}, params={num_params:,}) ===")
    for snr in snr_levels:
        test_ds = generate_mimo_dataset(
            num_samples=num_test_samples,
            nt=nt,
            nr=nr,
            tp=tp,
            snr_db_range=(float(snr), float(snr)),
            seed=20000 + int(snr * 10),
            compute_ls=False,
        )
        h_true = test_ds["H"].astype(np.complex128)
        pilots = test_ds["X"].astype(np.complex128)
        y_rx = test_ds["Y"].astype(np.complex128)
        noise_var = test_ds["noise_variance"].astype(np.float64)
        inputs = test_ds["inputs"]

        # Estimators + inference timing on identical batch
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

        mse_ls = compute_mse(h_true, h_ls)
        mse_mmse = compute_mse(h_true, h_mmse)
        mse_ai = compute_mse(h_true, h_ai)

        nmse_ls, nmse_ls_db = compute_nmse(h_true, h_ls)
        nmse_mmse, nmse_mmse_db = compute_nmse(h_true, h_mmse)
        nmse_ai, nmse_ai_db = compute_nmse(h_true, h_ai)

        ber_samples = min(DEFAULT_EXPERIMENT.ber_samples_per_snr, num_test_samples)
        ber_perfect = simulate_mimo_ber(
            h_true[:ber_samples],
            h_true[:ber_samples],
            snr_db=float(snr),
            num_data_symbols=DEFAULT_EXPERIMENT.ber_num_data_symbols,
            detector=DEFAULT_EXPERIMENT.detector,
            seed=30000 + int(snr * 10),
        )
        ber_ls = simulate_mimo_ber(
            h_true[:ber_samples],
            h_ls[:ber_samples],
            snr_db=float(snr),
            num_data_symbols=DEFAULT_EXPERIMENT.ber_num_data_symbols,
            detector=DEFAULT_EXPERIMENT.detector,
            seed=30000 + int(snr * 10),
        )
        ber_mmse = simulate_mimo_ber(
            h_true[:ber_samples],
            h_mmse[:ber_samples],
            snr_db=float(snr),
            num_data_symbols=DEFAULT_EXPERIMENT.ber_num_data_symbols,
            detector=DEFAULT_EXPERIMENT.detector,
            seed=30000 + int(snr * 10),
        )
        ber_ai = simulate_mimo_ber(
            h_true[:ber_samples],
            h_ai[:ber_samples],
            snr_db=float(snr),
            num_data_symbols=DEFAULT_EXPERIMENT.ber_num_data_symbols,
            detector=DEFAULT_EXPERIMENT.detector,
            seed=30000 + int(snr * 10),
        )

        ls_nmse_db_list.append(nmse_ls_db)
        mmse_nmse_db_list.append(nmse_mmse_db)
        ai_nmse_db_list.append(nmse_ai_db)

        ls_mse_list.append(mse_ls)
        mmse_mse_list.append(mse_mmse)
        ai_mse_list.append(mse_ai)

        ls_ber_list.append(ber_ls)
        mmse_ber_list.append(ber_mmse)
        ai_ber_list.append(ber_ai)
        perfect_ber_list.append(ber_perfect)

        ls_time_list.append(t_ls_ms)
        mmse_time_list.append(t_mmse_ms)
        ai_time_list.append(t_ai_ms)

        row = {
            "snr_db": snr,
            "ls_mse": mse_ls,
            "mmse_mse": mse_mmse,
            "ai_dnn_mse": mse_ai,
            "ls_nmse_lin": nmse_ls,
            "mmse_nmse_lin": nmse_mmse,
            "ai_dnn_nmse_lin": nmse_ai,
            "ls_nmse_db": nmse_ls_db,
            "mmse_nmse_db": nmse_mmse_db,
            "ai_dnn_nmse_db": nmse_ai_db,
            "perfect_csi_ber": ber_perfect,
            "ls_ber": ber_ls,
            "mmse_ber": ber_mmse,
            "ai_dnn_ber": ber_ai,
            "ls_time_ms": t_ls_ms,
            "mmse_time_ms": t_mmse_ms,
            "ai_dnn_time_ms": t_ai_ms,
            "ai_dnn_params": num_params,
        }
        rows.append(row)
        print(
            f"SNR={snr:4.0f} dB | NMSE(dB): LS={nmse_ls_db:7.2f}, MMSE={nmse_mmse_db:7.2f}, AI={nmse_ai_db:7.2f} "
            f"| BER: Perf={ber_perfect:.5f}, LS={ber_ls:.5f}, MMSE={ber_mmse:.5f}, AI={ber_ai:.5f}"
        )

    if save_outputs:
        save_csv(DEFAULT_PATHS.tables_dir / "snr_comparison.csv", rows)
        plot_metric_vs_x(
            snr_levels,
            {"LS": ls_nmse_db_list, "MMSE": mmse_nmse_db_list, "AI-DNN": ai_nmse_db_list},
            xlabel="SNR (dB)",
            ylabel="NMSE (dB)",
            title=f"NMSE vs SNR (Massive MIMO Nt={nt}, Nr={nr}, Tp={tp})",
            output_path=DEFAULT_PATHS.plots_dir / "nmse_vs_snr.png",
            logy=False,
        )
        plot_metric_vs_x(
            snr_levels,
            {"LS": ls_mse_list, "MMSE": mmse_mse_list, "AI-DNN": ai_mse_list},
            xlabel="SNR (dB)",
            ylabel="MSE",
            title=f"MSE vs SNR (Massive MIMO Nt={nt}, Nr={nr}, Tp={tp})",
            output_path=DEFAULT_PATHS.plots_dir / "mse_vs_snr.png",
            logy=True,
        )
        plot_metric_vs_x(
            snr_levels,
            {
                "Perfect CSI": perfect_ber_list,
                "LS": ls_ber_list,
                "MMSE": mmse_ber_list,
                "AI-DNN": ai_ber_list,
            },
            xlabel="SNR (dB)",
            ylabel="Bit Error Rate (BER)",
            title=f"Downlink RZF Precoded BER vs SNR (Nt={nt}, Nr={nr}, BPSK)",
            output_path=DEFAULT_PATHS.plots_dir / "ber_vs_snr.png",
            logy=True,
        )
        plot_inference_time_bar(
            ["LS", "MMSE", "AI-DNN"],
            [float(np.mean(ls_time_list)), float(np.mean(mmse_time_list)), float(np.mean(ai_time_list))],
            output_path=DEFAULT_PATHS.plots_dir / "inference_time_comparison.png",
            title=f"Batch Inference Time ({num_test_samples} samples, Nt={nt}, Nr={nr}, Tp={tp})",
        )
        # Also refresh root legacy plot names so existing references stay up to date
        plot_metric_vs_x(
            snr_levels,
            {"LS": ls_mse_list, "MMSE": mmse_mse_list, "AI-DNN": ai_mse_list},
            xlabel="SNR (dB)",
            ylabel="MSE",
            title=f"MSE vs SNR (Massive MIMO Nt={nt}, Nr={nr}, Tp={tp})",
            output_path=DEFAULT_PATHS.plots_dir.parent.parent / "mse_comparison.png",
            logy=True,
        )
        plot_metric_vs_x(
            snr_levels,
            {"Perfect CSI": perfect_ber_list, "LS": ls_ber_list, "MMSE": mmse_ber_list, "AI-DNN": ai_ber_list},
            xlabel="SNR (dB)",
            ylabel="Bit Error Rate (BER)",
            title=f"BER vs SNR (Massive MIMO Nt={nt}, Nr={nr}, Tp={tp})",
            output_path=DEFAULT_PATHS.plots_dir.parent.parent / "ber_comparison.png",
            logy=True,
        )

    return rows


if __name__ == "__main__":
    run_snr_experiment()
