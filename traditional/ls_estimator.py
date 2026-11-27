"""Least Squares (LS) channel estimators for SISO and batched Massive MIMO systems."""

from typing import Union
import numpy as np


def mimo_ls_estimate(
    received_pilots: np.ndarray,
    transmitted_pilots: np.ndarray,
    orthogonal_atol: float = 1e-6,
) -> np.ndarray:
    """Estimate complex MIMO channel matrix H using Least Squares (LS).

    System model:
        Y = H @ X + N
    where:
        Y has shape (Nr, Tp) or (B, Nr, Tp)
        X has shape (Nt, Tp) or (B, Nt, Tp)
        H_LS has shape (Nr, Nt) or (B, Nr, Nt)

    Formula:
        H_LS = Y @ X^H @ (X @ X^H)^{-1}

    Special cases:
        1. Orthogonal pilots satisfying X @ X^H = Tp * I_{Nt}:
           H_LS = (Y @ X^H) / Tp
        2. Underdetermined pilots (Tp < Nt) where X @ X^H is singular:
           Uses minimum-norm Moore-Penrose pseudoinverse H_LS = Y @ pinv(X).
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

    nt, tp = x.shape[-2], x.shape[-1]

    # Fast exact path for shared 2D pilot matrix X of shape (Nt, Tp)
    if x.ndim == 2:
        x_h = x.conj().T  # (Tp, Nt)
        gram = x @ x_h    # (Nt, Nt)
        eye_nt = np.eye(nt, dtype=gram.dtype)

        if tp >= nt and np.max(np.abs(gram - tp * eye_nt)) <= orthogonal_atol:
            h_est = (y @ x_h) / float(tp)
        elif tp >= nt and np.linalg.cond(gram) < 1e10:
            # Solve (Y @ X^H) @ inv(gram) via transpose: gram^T @ H^T = (Y @ X^H)^T
            yx_h = y @ x_h  # (B, Nr, Nt)
            h_est = np.linalg.solve(gram.T, yx_h.swapaxes(-1, -2)).swapaxes(-1, -2)
        else:
            # Underdetermined (Tp < Nt) or ill-conditioned: minimum-norm pseudoinverse
            x_pinv = np.linalg.pinv(x)  # (Tp, Nt)
            h_est = y @ x_pinv
    else:
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Batch size mismatch: Y has {y.shape[0]}, X has {x.shape[0]}")
        x_h = np.conj(np.swapaxes(x, -1, -2))  # (B, Tp, Nt)
        gram = x @ x_h  # (B, Nt, Nt)
        yx_h = y @ x_h  # (B, Nr, Nt)
        if tp >= nt:
            try:
                h_est = np.linalg.solve(
                    np.swapaxes(gram, -1, -2),
                    np.swapaxes(yx_h, -1, -2),
                ).swapaxes(-1, -2)
            except np.linalg.LinAlgError:
                h_est = y @ np.linalg.pinv(x)
        else:
            h_est = y @ np.linalg.pinv(x)

    return h_est[0] if squeeze_batch else h_est


def ls_channel_estimation(
    received_pilots: np.ndarray,
    transmitted_pilots: np.ndarray,
) -> Union[np.ndarray, complex]:
    """Backward-compatible LS estimator supporting both 1D SISO and 2D/3D Massive MIMO."""
    y = np.asarray(received_pilots)
    x = np.asarray(transmitted_pilots)

    if y.ndim == 1 and x.ndim == 1:
        numerator = np.sum(np.conj(x) * y)
        denominator = np.sum(np.abs(x) ** 2)
        if denominator == 0:
            raise ValueError("Pilot symbols cannot all be zero")
        return numerator / denominator

    return mimo_ls_estimate(y, x)
