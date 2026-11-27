"""Minimum Mean Square Error (MMSE) channel estimator for Massive MIMO systems."""

from typing import Union
import numpy as np


def mimo_mmse_estimate(
    received_pilots: np.ndarray,
    transmitted_pilots: np.ndarray,
    noise_variance: Union[float, np.ndarray],
) -> np.ndarray:
    """Estimate complex MIMO channel matrix H using linear MMSE under i.i.d. Rayleigh prior.

    Assumptions:
        - H_{i,j} ~ CN(0, 1) i.i.d. (so row covariance R_h = I_{Nt}).
        - N_{i,k} ~ CN(0, sigma^2) i.i.d., where sigma^2 = noise_variance.
        - System model: Y = H @ X + N, with Y: (..., Nr, Tp), X: (..., Nt, Tp).

    Exact LMMSE Estimator:
        H_MMSE = Y @ X^H @ (X @ X^H + sigma^2 * I_{Nt})^{-1}

    Supports:
        - Single matrix Y: (Nr, Tp), X: (Nt, Tp), scalar noise_variance
        - Batch Y: (B, Nr, Tp), shared X: (Nt, Tp), scalar or (B,) noise_variance
        - Batch Y: (B, Nr, Tp), per-sample X: (B, Nt, Tp), scalar or (B,) noise_variance
    """
    y = np.asarray(received_pilots)
    x = np.asarray(transmitted_pilots)

    if y.ndim not in (2, 3):
        raise ValueError(f"Expected Y to be 2D (Nr, Tp) or 3D (B, Nr, Tp), got shape {y.shape}")
    if x.ndim not in (2, 3):
        raise ValueError(f"Expected X to be 2D (Nt, Tp) or 3D (B, Nt, Tp), got shape {x.shape}")
    if y.shape[-1] != x.shape[-1]:
        raise ValueError(
            f"Pilot length mismatch: Y has Tp={y.shape[-1]}, X has Tp={x.shape[-1]}"
        )

    squeeze_batch = False
    if y.ndim == 2:
        y = y[None, ...]
        squeeze_batch = True

    batch_size = y.shape[0]
    nt = x.shape[-2]

    sigma_sq = np.asarray(noise_variance, dtype=np.float64)
    if sigma_sq.ndim == 0:
        if float(sigma_sq) < 0:
            raise ValueError(f"noise_variance must be non-negative, got {float(sigma_sq)}")
    else:
        sigma_sq = sigma_sq.reshape(-1)
        if sigma_sq.shape[0] != batch_size:
            raise ValueError(
                f"noise_variance length ({sigma_sq.shape[0]}) must match batch size ({batch_size})"
            )
        if np.any(sigma_sq < 0):
            raise ValueError("noise_variance entries must be non-negative")

    if x.ndim == 2:
        x_h = x.conj().T  # (Tp, Nt)
        gram = x @ x_h    # (Nt, Nt)
        yx_h = y @ x_h    # (B, Nr, Nt)
        eye_nt = np.eye(nt, dtype=gram.dtype)

        if sigma_sq.ndim == 0:
            reg = gram + float(sigma_sq) * eye_nt  # (Nt, Nt)
            # H_MMSE = (Y X^H) @ inv(reg)  <=>  reg^T @ H^T = (Y X^H)^T
            h_est = np.linalg.solve(reg.T, yx_h.swapaxes(-1, -2)).swapaxes(-1, -2)
        else:
            reg = gram[None, :, :] + sigma_sq[:, None, None] * eye_nt[None, :, :]  # (B, Nt, Nt)
            h_est = np.linalg.solve(
                np.swapaxes(reg, -1, -2),
                np.swapaxes(yx_h, -1, -2),
            ).swapaxes(-1, -2)
    else:
        if x.shape[0] != batch_size:
            raise ValueError(f"Batch size mismatch: Y has {batch_size}, X has {x.shape[0]}")
        x_h = np.conj(np.swapaxes(x, -1, -2))
        gram = x @ x_h
        yx_h = y @ x_h
        eye_nt = np.eye(nt, dtype=gram.dtype)[None, :, :]
        if sigma_sq.ndim == 0:
            reg = gram + float(sigma_sq) * eye_nt
        else:
            reg = gram + sigma_sq[:, None, None] * eye_nt
        h_est = np.linalg.solve(
            np.swapaxes(reg, -1, -2),
            np.swapaxes(yx_h, -1, -2),
        ).swapaxes(-1, -2)

    return h_est[0] if squeeze_batch else h_est
