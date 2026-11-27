"""Experiments 3 & 5: SNR Generalization (Interpolation/Extrapolation) & Cross-Antenna Analysis."""

from typing import Dict, List
import numpy as np
import torch

from config.config import DEFAULT_EXPERIMENT, DEFAULT_MIMO, DEFAULT_PATHS
from data.dataset import generate_mimo_dataset
from evaluation.metrics import compute_nmse
from evaluation.visualization import plot_metric_vs_x, save_csv
from traditional.ls_estimator import mimo_ls_estimate
from traditional.mmse_estimator import mimo_mmse_estimate
from train import load_trained_estimator, predict_mimo_channel, train_channel_estimator


def run_snr_generalization_experiment(
    nt: int = DEFAULT_MIMO.nt,
    nr: int = DEFAULT_MIMO.nr,
    tp: int = DEFAULT_MIMO.tp,
    train_samples: int = DEFAULT_EXPERIMENT.sweep_train_samples,
    test_samples: int = DEFAULT_EXPERIMENT.sweep_test_samples,
    epochs: int = DEFAULT_EXPERIMENT.sweep_epochs,
) -> List[Dict[str, object]]:
    """Experiment 3: Separate SNR interpolation and SNR extrapolation tests.

    1. Interpolation model: trained on [0, 10] U [20, 30] dB, tested across 0..30 dB
       including unseen gap SNRs [12.5, 15.0, 17.5] dB.
    2. Extrapolation model: trained ONLY on [5, 20] dB, tested across 0..30 dB
       including unseen lower (0 dB) and higher (25, 30 dB) SNRs.
    """
    DEFAULT_PATHS.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Interpolation dataset: half from [0, 10] dB, half from [20, 30] dB
    half_n = train_samples // 2
    ds_low = generate_mimo_dataset(half_n, nt=nt, nr=nr, tp=tp, snr_db_range=(0.0, 10.0), seed=8001, compute_ls=False)
    ds_high = generate_mimo_dataset(train_samples - half_n, nt=nt, nr=nr, tp=tp, snr_db_range=(20.0, 30.0), seed=8002, compute_ls=False)
    interp_inputs = np.concatenate([ds_low["inputs"], ds_high["inputs"]], axis=0)
    interp_targets = np.concatenate([ds_low["targets"], ds_high["targets"]], axis=0)

    interp_res = train_channel_estimator(
        inputs=interp_inputs,
        targets=interp_targets,
        nr=nr,
        nt=nt,
        tp=tp,
        checkpoint_path=DEFAULT_PATHS.checkpoints_dir / f"exp_snr_interp_nt{nt}.pth",
        epochs=epochs,
        seed=81,
        backup_existing=False,
        verbose=False,
    )
    interp_model = interp_res["model"].to(device)
    interp_mean = torch.from_numpy(interp_res["input_mean"]).to(device)
    interp_std = torch.from_numpy(interp_res["input_std"]).to(device)

    # 2. Extrapolation dataset: trained ONLY on [5, 20] dB
    ds_mid = generate_mimo_dataset(train_samples, nt=nt, nr=nr, tp=tp, snr_db_range=(5.0, 20.0), seed=8003, compute_ls=False)
    extrap_res = train_channel_estimator(
        inputs=ds_mid["inputs"],
        targets=ds_mid["targets"],
        nr=nr,
        nt=nt,
        tp=tp,
        checkpoint_path=DEFAULT_PATHS.checkpoints_dir / f"exp_snr_extrap_nt{nt}.pth",
        epochs=epochs,
        seed=82,
        backup_existing=False,
        verbose=False,
    )
    extrap_model = extrap_res["model"].to(device)
    extrap_mean = torch.from_numpy(extrap_res["input_mean"]).to(device)
    extrap_std = torch.from_numpy(extrap_res["input_std"]).to(device)

    # 3. Full-range baseline model
    full_ckpt = DEFAULT_PATHS.checkpoints_dir / f"mimo_dnn_nt{nt}_nr{nr}_tp{tp}.pth"
    full_model, full_mean, full_std, _ = load_trained_estimator(full_ckpt, device=device)

    eval_snrs = [0.0, 5.0, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0]
    rows: List[Dict[str, object]] = []
    mmse_list, full_list, interp_list, extrap_list = [], [], [], []

    print(f"\n=== EXPERIMENT 3: SNR Interpolation & Extrapolation (Nt={nt}, Nr={nr}, Tp={tp}) ===")
    for snr in eval_snrs:
        test_ds = generate_mimo_dataset(
            num_samples=test_samples,
            nt=nt,
            nr=nr,
            tp=tp,
            snr_db_range=(snr, snr),
            seed=9000 + int(snr * 10),
            compute_ls=False,
        )
        h_true = test_ds["H"].astype(np.complex128)
        h_mmse = mimo_mmse_estimate(test_ds["Y"].astype(np.complex128), test_ds["X"].astype(np.complex128), test_ds["noise_variance"].astype(np.float64))
        h_full = predict_mimo_channel(full_model, test_ds["inputs"], full_mean, full_std, nr=nr, nt=nt, device=device)
        h_interp = predict_mimo_channel(interp_model, test_ds["inputs"], interp_mean, interp_std, nr=nr, nt=nt, device=device)
        h_extrap = predict_mimo_channel(extrap_model, test_ds["inputs"], extrap_mean, extrap_std, nr=nr, nt=nt, device=device)

        _, nmse_mmse_db = compute_nmse(h_true, h_mmse)
        _, nmse_full_db = compute_nmse(h_true, h_full)
        _, nmse_interp_db = compute_nmse(h_true, h_interp)
        _, nmse_extrap_db = compute_nmse(h_true, h_extrap)

        mmse_list.append(nmse_mmse_db)
        full_list.append(nmse_full_db)
        interp_list.append(nmse_interp_db)
        extrap_list.append(nmse_extrap_db)

        interp_status = "unseen_interp" if (10.0 < snr < 20.0) else "seen"
        extrap_status = "unseen_extrap" if (snr < 5.0 or snr > 20.0) else "seen"
        rows.append(
            {
                "snr_db": snr,
                "interp_regime": interp_status,
                "extrap_regime": extrap_status,
                "mmse_nmse_db": nmse_mmse_db,
                "ai_full_0_30_nmse_db": nmse_full_db,
                "ai_interp_nmse_db": nmse_interp_db,
                "ai_extrap_nmse_db": nmse_extrap_db,
            }
        )
        print(
            f"SNR={snr:4.1f} dB | MMSE={nmse_mmse_db:6.2f} | Full={nmse_full_db:6.2f} | "
            f"Interp({interp_status:13s})={nmse_interp_db:6.2f} | Extrap({extrap_status:13s})={nmse_extrap_db:6.2f}"
        )

    save_csv(DEFAULT_PATHS.tables_dir / "snr_generalization.csv", rows)
    plot_metric_vs_x(
        eval_snrs,
        {
            "MMSE (Oracle SNR)": mmse_list,
            "AI-DNN (Trained 0-30 dB)": full_list,
            "AI-DNN Interp (Trained 0-10 & 20-30 dB)": interp_list,
            "AI-DNN Extrap (Trained 5-20 dB)": extrap_list,
        },
        xlabel="SNR (dB)",
        ylabel="NMSE (dB)",
        title=f"SNR Generalization: Interpolation vs Extrapolation (Nt={nt}, Nr={nr})",
        output_path=DEFAULT_PATHS.plots_dir / "snr_generalization.png",
        logy=False,
    )
    return rows


def run_cross_antenna_generalization_experiment(
    source_nt: int = 32,
    target_nt: int = 16,
    nr: int = 4,
    eval_snr_db: float = 15.0,
    test_samples: int = 2000,
) -> List[Dict[str, object]]:
    """Experiment 5: Cross-antenna generalization analysis for fixed-dimension DNN.

    A fixed-input/output DNN trained for (Nt=32, Nr=4, Tp=32) expects input_dim=256 and
    output_dim=256. It cannot natively process arbitrary antenna counts (e.g. Nt=16, input_dim=128)
    without padding/masking or retraining. Here we test zero-padded sub-array evaluation
    against a matched Nt=16 DNN and analytical LS/MMSE estimators to quantify the mismatch.
    """
    DEFAULT_PATHS.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_ckpt = DEFAULT_PATHS.checkpoints_dir / f"mimo_dnn_nt{source_nt}_nr{nr}_tp{source_nt}.pth"
    src_model, src_mean, src_std, _ = load_trained_estimator(src_ckpt, device=device)

    matched_ckpt = DEFAULT_PATHS.checkpoints_dir / f"exp_ant_nt{target_nt}_nr{nr}_tp{target_nt}.pth"
    if not matched_ckpt.exists():
        # Train matched target_nt model if antenna_experiment wasn't run first
        ds_match = generate_mimo_dataset(
            DEFAULT_EXPERIMENT.sweep_train_samples,
            nt=target_nt,
            nr=nr,
            tp=target_nt,
            snr_db_range=DEFAULT_MIMO.snr_db_range,
            seed=9500,
            compute_ls=False,
        )
        train_channel_estimator(
            ds_match["inputs"],
            ds_match["targets"],
            nr=nr,
            nt=target_nt,
            tp=target_nt,
            checkpoint_path=matched_ckpt,
            epochs=DEFAULT_EXPERIMENT.sweep_epochs,
            seed=95,
            backup_existing=False,
            verbose=False,
        )
    matched_model, matched_mean, matched_std, _ = load_trained_estimator(matched_ckpt, device=device)

    # Generate Nt=16 test data and evaluate matched vs zero-padded cross-antenna transfer
    test_ds = generate_mimo_dataset(
        num_samples=test_samples,
        nt=target_nt,
        nr=nr,
        tp=target_nt,
        snr_db_range=(eval_snr_db, eval_snr_db),
        seed=9600,
        compute_ls=False,
    )
    h_true = test_ds["H"].astype(np.complex128)
    y_rx = test_ds["Y"].astype(np.complex128)
    x_pil = test_ds["X"].astype(np.complex128)
    noise_var = test_ds["noise_variance"].astype(np.float64)

    h_ls = mimo_ls_estimate(y_rx, x_pil)
    h_mmse = mimo_mmse_estimate(y_rx, x_pil, noise_var)
    h_matched = predict_mimo_channel(matched_model, test_ds["inputs"], matched_mean, matched_std, nr=nr, nt=target_nt, device=device)

    # Zero-pad Y from (B, Nr, 16) -> (B, Nr, 32) to feed into Nt=32 DNN and slice first 16 TX columns
    y_padded = np.zeros((test_samples, nr, source_nt), dtype=np.complex64)
    y_padded[:, :, :target_nt] = y_rx.astype(np.complex64)
    padded_inputs = np.concatenate(
        [y_padded.real.reshape(test_samples, -1), y_padded.imag.reshape(test_samples, -1)], axis=1
    ).astype(np.float32)
    h_src_padded = predict_mimo_channel(src_model, padded_inputs, src_mean, src_std, nr=nr, nt=source_nt, device=device)
    h_cross_sliced = h_src_padded[:, :, :target_nt]

    rows = []
    for name, est in [
        ("LS (Matched Nt=16)", h_ls),
        ("MMSE (Matched Nt=16)", h_mmse),
        ("AI-DNN (Matched Nt=16)", h_matched),
        (f"AI-DNN (Trained Nt={source_nt}, zero-pad to Nt={target_nt})", h_cross_sliced),
    ]:
        nmse_lin, nmse_db = compute_nmse(h_true, est)
        rows.append(
            {
                "method": name,
                "source_nt": source_nt if "Trained" in name else target_nt,
                "target_nt": target_nt,
                "nr": nr,
                "eval_snr_db": eval_snr_db,
                "nmse_lin": nmse_lin,
                "nmse_db": nmse_db,
            }
        )

    print(f"\n=== EXPERIMENT 5: Cross-Antenna Generalization (Target Nt={target_nt}, SNR={eval_snr_db} dB) ===")
    for r in rows:
        print(f"  {r['method']:55s} | NMSE = {r['nmse_db']:6.2f} dB")

    save_csv(DEFAULT_PATHS.tables_dir / "cross_antenna_generalization.csv", rows)
    return rows


if __name__ == "__main__":
    run_snr_generalization_experiment()
    run_cross_antenna_generalization_experiment()
