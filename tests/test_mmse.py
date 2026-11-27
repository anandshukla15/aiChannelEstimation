"""Unit tests for MMSE channel estimation against manual examples and theoretical properties."""

import numpy as np
import pytest

from traditional.ls_estimator import mimo_ls_estimate
from traditional.mmse_estimator import mimo_mmse_estimate
from utils.channel import mimo_rayleigh_channel
from utils.noise import generate_complex_awgn
from utils.pilots import generate_hadamard_pilots


def test_manual_2x2_mmse_matches_explicit_inverse():
    Y = np.array([[2.0 - 1.0j, 0.5 + 1.5j], [-1.0 + 0.2j, 1.2 - 0.8j]], dtype=np.complex128)
    X = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    sigma_sq = 2.0

    # Hand calculation: X @ X^H = 2 * I_2 => inv(2*I + 2*I) = 0.25 * I_2
    # So H_MMSE = 0.25 * Y @ X^H = 0.5 * H_LS
    expected = Y @ X.conj().T @ np.linalg.inv(X @ X.conj().T + sigma_sq * np.eye(2))
    actual = mimo_mmse_estimate(Y, X, noise_variance=sigma_sq)
    assert actual.shape == (2, 2)
    assert np.allclose(actual, expected, atol=1e-12)
    assert np.allclose(actual, 0.5 * mimo_ls_estimate(Y, X), atol=1e-12)


def test_batched_mmse_with_per_sample_noise_variance():
    rng = np.random.default_rng(11)
    B, Nr, Nt, Tp = 10, 4, 8, 8
    X = generate_hadamard_pilots(Nt, Tp)
    H = mimo_rayleigh_channel(B, Nr, Nt, rng=rng)
    sigma_sq = rng.uniform(0.1, 5.0, size=B)
    N = generate_complex_awgn((B, Nr, Tp), sigma_sq[:, None, None], rng=rng)
    Y = H @ X + N

    H_mmse_batch = mimo_mmse_estimate(Y, X, sigma_sq)
    assert H_mmse_batch.shape == (B, Nr, Nt)
    for i in range(B):
        H_mmse_single = mimo_mmse_estimate(Y[i], X, float(sigma_sq[i]))
        assert np.allclose(H_mmse_batch[i], H_mmse_single, atol=1e-12)


def test_mmse_outperforms_ls_under_rayleigh_prior():
    rng = np.random.default_rng(2025)
    B, Nr, Nt, Tp = 1000, 4, 16, 16
    X = generate_hadamard_pilots(Nt, Tp)
    H = mimo_rayleigh_channel(B, Nr, Nt, rng=rng)
    # SNR = 0 dB -> sigma^2 = Nt = 16
    sigma_sq = 16.0
    N = generate_complex_awgn((B, Nr, Tp), sigma_sq, rng=rng)
    Y = H @ X + N

    h_ls = mimo_ls_estimate(Y, X)
    h_mmse = mimo_mmse_estimate(Y, X, sigma_sq)

    nmse_ls = np.mean(np.abs(H - h_ls) ** 2) / np.mean(np.abs(H) ** 2)
    nmse_mmse = np.mean(np.abs(H - h_mmse) ** 2) / np.mean(np.abs(H) ** 2)

    # Theoretical at 0 dB (sigma^2 = Tp): LS NMSE = 1.0, MMSE NMSE = 0.5
    assert 0.9 < nmse_ls < 1.1
    assert 0.45 < nmse_mmse < 0.55
    assert nmse_mmse < nmse_ls


def test_negative_noise_variance_raises():
    Y = np.zeros((2, 4), dtype=np.complex128)
    X = np.eye(4, dtype=np.complex128)
    with pytest.raises(ValueError, match="non-negative"):
        mimo_mmse_estimate(Y, X, -0.5)
