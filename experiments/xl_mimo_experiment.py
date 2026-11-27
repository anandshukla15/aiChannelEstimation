"""Experiment 6 (6G Research Extension): XL-MIMO Near-Field Spherical-Wave Channel Estimation.

Investigates 6G Extremely Large-Scale MIMO (XL-MIMO) effects at 28 GHz:
    - Carrier frequency f_c = 28 GHz (lambda = 10.7 mm)
    - ULA with Nt = 64 elements, spacing d = lambda / 2 = 5.35 mm, aperture D = 0.337 m
    - Rayleigh (Fraunhofer) distance d_R = 2 * D^2 / lambda = 21.25 m
    - Near-field regime: r in [3, 20] m < d_R (spherical wavefront curvature + spatial non-stationary visibility)
    - Far-field regime: r = 200 m >> d_R (planar wavefront)
    - Baseline: i.i.d. Rayleigh fading

Demonstrates:
    1. Conventional i.i.d. MMSE is mismatched on physically structured XL-MIMO near-field channels.
    2. An AI-DNN trained on near-field XL-MIMO channels exploits spherical-wave & visibility structure
       and outperforms both LS and mismatched i.i.d. MMSE.
    3. A Rayleigh-trained AI-DNN suffers domain shift when evaluated on near-field XL-MIMO channels.
"""

from typing import Dict, List
import numpy as np
import torch

from config.config import DEFAULT_EXPERIMENT, DEFAULT_MIMO, DEFAULT_PATHS, DEFAULT_XL_MIMO
from data.dataset import generate_mimo_dataset
from evaluation.metrics import compute_nmse
from evaluation.visualization import plot_metric_vs_x, save_csv
from traditional.ls_estimator import mimo_ls_estimate
from traditional.mmse_estimator import mimo_mmse_estimate
from train import predict_mimo_channel, train_channel_estimator


def run_xl_mimo_experiment(
    snr_levels: List[float] = None,
    train_samples: int = DEFAULT_XL_MIMO.num_train_samples,
    test_samples: int = DEFAULT_XL_MIMO.num_test_samples,
    epochs: int = DEFAULT_EXPERIMENT.sweep_epochs,
) -> List[Dict[str, object]]:
    DEFAULT_PATHS.ensure_dirs()
    if snr_levels is None:
        snr_levels = list(DEFAULT_EXPERIMENT.snr_levels)

    cfg = DEFAULT_XL_MIMO
    nt, nr, tp = cfg.nt, cfg.nr, cfg.tp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=== 6G XL-MIMO EXTENSION: Near-Field Spherical-Wave vs Far-Field & Rayleigh ===")
    print(
        f"f_c={cfg.carrier_frequency_hz/1e9:.1f} GHz, lambda={cfg.wavelength_m*1e3:.2f} mm, "
        f"d={cfg.antenna_spacing_m*1e3:.2f} mm, Nt={nt}, Aperture D={cfg.tx_aperture_m:.3f} m, "
        f"Rayleigh Distance d_R={cfg.rayleigh_distance_m:.2f} m"
    )

    # 1. Train AI-DNN on Near-Field XL-MIMO dataset
    nf_train = generate_mimo_dataset(
        num_samples=train_samples,
        nt=nt,
        nr=nr,
        tp=tp,
        snr_db_range=DEFAULT_MIMO.snr_db_range,
        channel_model="xl_mimo",
        xl_regime="near_field",
        seed=11001,
        compute_ls=False,
    )
    nf_res = train_channel_estimator(
        inputs=nf_train["inputs"],
        targets=nf_train["targets"],
        nr=nr,
        nt=nt,
        tp=tp,
        checkpoint_path=DEFAULT_PATHS.checkpoints_dir / f"exp_xl_nearfield_nt{nt}.pth",
        epochs=epochs,
        seed=111,
        backup_existing=False,
        verbose=False,
    )
    nf_model = nf_res["model"].to(device)
    nf_mean = torch.from_numpy(nf_res["input_mean"]).to(device)
    nf_std = torch.from_numpy(nf_res["input_std"]).to(device)

    # 2. Train AI-DNN on Far-Field XL-MIMO dataset
    ff_train = generate_mimo_dataset(
        num_samples=train_samples,
        nt=nt,
        nr=nr,
        tp=tp,
        snr_db_range=DEFAULT_MIMO.snr_db_range,
        channel_model="xl_mimo",
        xl_regime="far_field",
        seed=11002,
        compute_ls=False,
    )
    ff_res = train_channel_estimator(
        inputs=ff_train["inputs"],
        targets=ff_train["targets"],
        nr=nr,
        nt=nt,
        tp=tp,
        checkpoint_path=DEFAULT_PATHS.checkpoints_dir / f"exp_xl_farfield_nt{nt}.pth",
        epochs=epochs,
        seed=112,
        backup_existing=False,
        verbose=False,
    )
    ff_model = ff_res["model"].to(device)
    ff_mean = torch.from_numpy(ff_res["input_mean"]).to(device)
    ff_std = torch.from_numpy(ff_res["input_std"]).to(device)

    # 3. Train AI-DNN on i.i.d. Rayleigh Nt=64 baseline for cross-model comparison
    ray_train = generate_mimo_dataset(
        num_samples=train_samples,
        nt=nt,
        nr=nr,
        tp=tp,
        snr_db_range=DEFAULT_MIMO.snr_db_range,
        channel_model="rayleigh",
        seed=11003,
        compute_ls=False,
    )
    ray_res = train_channel_estimator(
        inputs=ray_train["inputs"],
        targets=ray_train["targets"],
        nr=nr,
        nt=nt,
        tp=tp,
        checkpoint_path=DEFAULT_PATHS.checkpoints_dir / f"exp_xl_rayleigh_nt{nt}.pth",
        epochs=epochs,
        seed=113,
        backup_existing=False,
        verbose=False,
    )
    ray_model = ray_res["model"].to(device)
    ray_mean = torch.from_numpy(ray_res["input_mean"]).to(device)
    ray_std = torch.from_numpy(ray_res["input_std"]).to(device)

    rows: List[Dict[str, object]] = []
    nf_ls_list, nf_mmse_list, nf_ai_list, nf_rayai_list, ff_ai_list, ray_mmse_list = [], [], [], [], [], []

    for snr in snr_levels:
        nf_test = generate_mimo_dataset(
            num_samples=test_samples,
            nt=nt,
            nr=nr,
            tp=tp,
            snr_db_range=(float(snr), float(snr)),
            channel_model="xl_mimo",
            xl_regime="near_field",
            seed=12000 + int(snr * 10),
            compute_ls=False,
        )
        ff_test = generate_mimo_dataset(
            num_samples=test_samples,
            nt=nt,
            nr=nr,
            tp=tp,
            snr_db_range=(float(snr), float(snr)),
            channel_model="xl_mimo",
            xl_regime="far_field",
            seed=13000 + int(snr * 10),
            compute_ls=False,
        )
        ray_test = generate_mimo_dataset(
            num_samples=test_samples,
            nt=nt,
            nr=nr,
            tp=tp,
            snr_db_range=(float(snr), float(snr)),
            channel_model="rayleigh",
            seed=14000 + int(snr * 10),
            compute_ls=False,
        )

        # Near-field evaluation
        h_nf = nf_test["H"].astype(np.complex128)
        h_nf_ls = mimo_ls_estimate(nf_test["Y"].astype(np.complex128), nf_test["X"].astype(np.complex128))
        h_nf_mmse = mimo_mmse_estimate(
            nf_test["Y"].astype(np.complex128), nf_test["X"].astype(np.complex128), nf_test["noise_variance"].astype(np.float64)
        )
        h_nf_ai = predict_mimo_channel(nf_model, nf_test["inputs"], nf_mean, nf_std, nr=nr, nt=nt, device=device)
        h_nf_ray_ai = predict_mimo_channel(ray_model, nf_test["inputs"], ray_mean, ray_std, nr=nr, nt=nt, device=device)

        # Far-field evaluation
        h_ff = ff_test["H"].astype(np.complex128)
        h_ff_mmse = mimo_mmse_estimate(
            ff_test["Y"].astype(np.complex128), ff_test["X"].astype(np.complex128), ff_test["noise_variance"].astype(np.float64)
        )
        h_ff_ai = predict_mimo_channel(ff_model, ff_test["inputs"], ff_mean, ff_std, nr=nr, nt=nt, device=device)

        # Rayleigh evaluation
        h_ray = ray_test["H"].astype(np.complex128)
        h_ray_mmse = mimo_mmse_estimate(
            ray_test["Y"].astype(np.complex128), ray_test["X"].astype(np.complex128), ray_test["noise_variance"].astype(np.float64)
        )

        _, nmse_nf_ls_db = compute_nmse(h_nf, h_nf_ls)
        _, nmse_nf_mmse_db = compute_nmse(h_nf, h_nf_mmse)
        _, nmse_nf_ai_db = compute_nmse(h_nf, h_nf_ai)
        _, nmse_nf_ray_ai_db = compute_nmse(h_nf, h_nf_ray_ai)
        _, nmse_ff_mmse_db = compute_nmse(h_ff, h_ff_mmse)
        _, nmse_ff_ai_db = compute_nmse(h_ff, h_ff_ai)
        _, nmse_ray_mmse_db = compute_nmse(h_ray, h_ray_mmse)

        nf_ls_list.append(nmse_nf_ls_db)
        nf_mmse_list.append(nmse_nf_mmse_db)
        nf_ai_list.append(nmse_nf_ai_db)
        nf_rayai_list.append(nmse_nf_ray_ai_db)
        ff_ai_list.append(nmse_ff_ai_db)
        ray_mmse_list.append(nmse_ray_mmse_db)

        rows.append(
            {
                "snr_db": snr,
                "near_field_ls_nmse_db": nmse_nf_ls_db,
                "near_field_iid_mmse_nmse_db": nmse_nf_mmse_db,
                "near_field_ai_dnn_nmse_db": nmse_nf_ai_db,
                "near_field_rayleigh_ai_mismatch_nmse_db": nmse_nf_ray_ai_db,
                "far_field_iid_mmse_nmse_db": nmse_ff_mmse_db,
                "far_field_ai_dnn_nmse_db": nmse_ff_ai_db,
                "rayleigh_mmse_nmse_db": nmse_ray_mmse_db,
            }
        )
        print(
            f"SNR={snr:4.0f} dB | NF-LS={nmse_nf_ls_db:6.2f} | NF-MMSE(iid)={nmse_nf_mmse_db:6.2f} | "
            f"NF-AI={nmse_nf_ai_db:6.2f} | NF-RayAI={nmse_nf_ray_ai_db:6.2f} | FF-AI={nmse_ff_ai_db:6.2f}"
        )

    save_csv(DEFAULT_PATHS.tables_dir / "xl_mimo_experiment.csv", rows)
    plot_metric_vs_x(
        snr_levels,
        {
            "Near-Field LS": nf_ls_list,
            "Near-Field Mismatched i.i.d. MMSE": nf_mmse_list,
            "Near-Field AI-DNN (XL-MIMO)": nf_ai_list,
            "Near-Field Rayleigh-Trained AI (Domain Shift)": nf_rayai_list,
            "Far-Field AI-DNN (Planar Wave)": ff_ai_list,
        },
        xlabel="SNR (dB)",
        ylabel="NMSE (dB)",
        title=f"6G XL-MIMO Near-Field vs Far-Field Estimation (Nt={nt}, f_c=28 GHz, d_R={cfg.rayleigh_distance_m:.1f} m)",
        output_path=DEFAULT_PATHS.plots_dir / "xl_mimo_near_vs_far_field.png",
        logy=False,
    )
    return rows


if __name__ == "__main__":
    run_xl_mimo_experiment()
