"""Unit tests for LS channel estimation against hand-calculated MIMO examples."""

import numpy as np
import pytest

from traditional.ls_estimator import ls_channel_estimation, mimo_ls_estimate
from utils.pilots import generate_hadamard_pilots


def test_siso_backward_compatibility():
    pilots = np.array([1.0, -1.0, 1.0, 1.0])
    h_true = 0.8 - 0.6j
    received = h_true * pilots
    h_est = ls_channel_estimation(received, pilots)
    assert np.isclose(h_est, h_true)


def test_manual_2x2_noiseless_ls():
    # Manually constructed 2x2 channel and orthogonal 2x2 Hadamard pilot matrix
    H = np.array([[1.0 + 2.0j, -0.5 + 0.5j], [0.2 - 1.1j, 1.4 + 0.0j]])
    X = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)  # X @ X^H = 2 * I_2
    Y = H @ X
    # Hand calculation: Y @ X^H / 2 == H
    H_ls = mimo_ls_estimate(Y, X)
    assert H_ls.shape == (2, 2)
    assert np.allclose(H_ls, H, atol=1e-12)


def test_manual_2x2_noisy_ls_exact_formula():
    # Verify H_LS = Y @ X^H @ inv(X @ X^H) for non-orthogonal invertible X
    H = np.array([[1.0 + 0.5j, -1.0 + 0.2j], [0.3 - 0.7j, 0.9 + 1.2j]])
    X = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.complex128)  # 2x3 non-orthogonal
    N = np.array([[0.1 - 0.1j, -0.2 + 0.05j, 0.0 + 0.2j], [0.05 + 0.1j, 0.1 - 0.1j, -0.05 + 0.0j]])
    Y = H @ X + N

    expected = Y @ X.conj().T @ np.linalg.inv(X @ X.conj().T)
    actual = mimo_ls_estimate(Y, X)
    assert np.allclose(actual, expected, atol=1e-12)


def test_batched_mimo_ls_matches_per_sample():
    rng = np.random.default_rng(99)
    B, Nr, Nt, Tp = 16, 4, 8, 8
    X = generate_hadamard_pilots(Nt, Tp)
    H = (rng.standard_normal((B, Nr, Nt)) + 1j * rng.standard_normal((B, Nr, Nt))) / np.sqrt(2.0)
    N = 0.1 * (rng.standard_normal((B, Nr, Tp)) + 1j * rng.standard_normal((B, Nr, Tp)))
    Y = H @ X + N

    H_ls_batch = mimo_ls_estimate(Y, X)
    assert H_ls_batch.shape == (B, Nr, Nt)
    for i in range(B):
        H_ls_single = mimo_ls_estimate(Y[i], X)
        assert np.allclose(H_ls_batch[i], H_ls_single, atol=1e-12)


def test_ls_shape_mismatch_raises():
    Y = np.zeros((4, 16), dtype=np.complex128)
    X = np.zeros((8, 12), dtype=np.complex128)
    with pytest.raises(ValueError, match="Pilot length mismatch"):
        mimo_ls_estimate(Y, X)
