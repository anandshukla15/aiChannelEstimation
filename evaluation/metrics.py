"""Performance metrics, MIMO BER communication simulation, and complexity measurement."""

import time
from typing import Callable, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn

from utils.modulation import bpsk_demodulate, bpsk_modulate, generate_bits
from utils.noise import compute_noise_variance, generate_complex_awgn


def compute_mse(h_true: np.ndarray, h_est: np.ndarray) -> float:
    """Compute elementwise Mean Squared Error: mean(|H - H_est|^2)."""
    err = np.asarray(h_true) - np.asarray(h_est)
    return float(np.mean(np.abs(err) ** 2))


def compute_nmse(h_true: np.ndarray, h_est: np.ndarray) -> Tuple[float, float]:
    """Compute Normalized Mean Squared Error (linear and dB).

    NMSE = E[||H - H_est||_F^2] / E[||H||_F^2]
    NMSE_dB = 10 * log10(NMSE)
    """
    h_t = np.asarray(h_true)
    h_e = np.asarray(h_est)
    num = float(np.mean(np.abs(h_t - h_e) ** 2))
    den = float(np.mean(np.abs(h_t) ** 2))
    if den == 0.0:
        raise ValueError("Reference channel has zero energy; NMSE is undefined.")
    nmse_lin = num / den
    nmse_db = float(10.0 * np.log10(max(nmse_lin, 1e-15)))
    return nmse_lin, nmse_db


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(
    fn: Callable[[], object],
    num_runs: int = 20,
    warmup_runs: int = 3,
) -> float:
    """Measure average wall-clock inference time in milliseconds."""
    for _ in range(warmup_runs):
        fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / num_runs) * 1000.0


def simulate_mimo_ber(
    h_true: np.ndarray,
    h_est: np.ndarray,
    snr_db: float,
    num_data_symbols: int = 50,
    detector: str = "mmse",
    seed: int = 2025,
) -> float:
    """Simulate end-to-end downlink/spatial MIMO data transmission and compute BER.

    Transmission & Detection Setup (documented assumptions):
        - Given H_true of shape (B, Nr, Nt) and estimated channel H_est of shape (B, Nr, Nt),
          with Nr <= Nt (Massive MIMO downlink: Nt BS antennas serving Nr spatial streams).
        - Transmits Ns = min(Nr, Nt) independent BPSK data streams per channel realization
          over `num_data_symbols` symbol intervals: S in {+1, -1}^{B x Ns x num_data_symbols}.
        - Uses linear transmit precoding constructed ONLY from the estimated channel H_est:
            * If detector == 'zf': W_raw = H_est^H (H_est H_est^H + 1e-8 I)^{-1}
            * If detector == 'mmse' (RZF): W_raw = H_est^H (H_est H_est^H + reg I)^{-1}
              where reg = Nr / SNR_linear.
        - Precoder is normalized to unit total transmit power per sample:
            W = W_raw / ||W_raw||_F  (so E[||W s||_2^2] = 1 per symbol slot when scaled by 1/sqrt(Ns),
            specifically W_norm = W_raw * sqrt(1.0 / trace(W_raw W_raw^H))).
        - Signal passes through the TRUE channel:
            Y_data = H_true @ (W_norm @ S) + N_data
          where N_data ~ CN(0, sigma_data^2) with sigma_data^2 = Nt / SNR_linear
          (consistent with pilot noise variance convention).
        - Receiver equalizes each stream using the known effective precoded channel gain
          G_est = H_est @ W_norm:
            S_hat = (G_est^H G_est + sigma_eff^2 I)^{-1} G_est^H Y_data (MMSE) or pinv(G_est) Y_data (ZF).
        - Hard-decision BPSK demodulation compares recovered bits against transmitted bits.
    """
    h_t = np.asarray(h_true, dtype=np.complex128)
    h_e = np.asarray(h_est, dtype=np.complex128)
    if h_t.shape != h_e.shape or h_t.ndim != 3:
        raise ValueError(f"Expected 3D arrays of matching shape (B, Nr, Nt), got {h_t.shape} and {h_e.shape}")

    batch_size, nr, nt = h_t.shape
    ns = min(nr, nt)
    rng = np.random.default_rng(seed)

    bits = generate_bits((batch_size, ns, num_data_symbols), rng=rng)
    symbols = bpsk_modulate(bits).astype(np.complex128) / np.sqrt(ns)  # total E[||s||^2] = 1

    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_var = compute_noise_variance(snr_db, signal_power=float(nt))

    eye_nr = np.eye(nr, dtype=np.complex128)[None, :, :]
    reg = (nr / snr_linear) if detector.lower() == "mmse" else 1e-8

    # Construct precoder from estimated channel H_est
    h_e_h = np.conj(np.swapaxes(h_e, -1, -2))  # (B, Nt, Nr)
    gram = h_e @ h_e_h + reg * eye_nr         # (B, Nr, Nr)
    # Solve gram @ inv_gram = I => W_raw^T = solve(...)
    inv_gram = np.linalg.solve(gram, np.broadcast_to(eye_nr, gram.shape))
    w_raw = h_e_h @ inv_gram                  # (B, Nt, Nr)
    if ns < nr:
        w_raw = w_raw[:, :, :ns]

    # Normalize precoder to unit Frobenius norm per sample so E[||W s||^2] = 1/Ns * ||W||_F^2 -> scale by sqrt(Ns)
    w_fro_sq = np.sum(np.abs(w_raw) ** 2, axis=(1, 2), keepdims=True)
    w_norm = w_raw * np.sqrt(float(ns) / np.maximum(w_fro_sq, 1e-12))

    # Transmit through TRUE channel H_true
    tx_signal = w_norm @ symbols  # (B, Nt, num_data_symbols)
    rx_clean = h_t @ tx_signal    # (B, Nr, num_data_symbols)
    rx_noise = generate_complex_awgn(rx_clean.shape, noise_var, rng=rng, dtype=np.complex128)
    rx_signal = rx_clean + rx_noise

    # Receiver equalization using estimated effective channel G_est = H_est @ W_norm
    g_est = h_e @ w_norm  # (B, Nr, Ns)
    g_est_h = np.conj(np.swapaxes(g_est, -1, -2))  # (B, Ns, Nr)
    eye_ns = np.eye(ns, dtype=np.complex128)[None, :, :]
    eq_reg = noise_var if detector.lower() == "mmse" else 1e-8
    eq_gram = g_est_h @ g_est + eq_reg * eye_ns
    eq_rhs = g_est_h @ rx_signal
    s_hat = np.linalg.solve(eq_gram, eq_rhs)

    detected_bits = bpsk_demodulate(s_hat)
    bit_errors = int(np.sum(detected_bits != bits))
    return float(bit_errors / bits.size)
