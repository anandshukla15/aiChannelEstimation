"""Unit tests for MSE, NMSE, parameter counting, and MIMO BER simulation."""

import numpy as np

from evaluation.metrics import (
    compute_mse,
    compute_nmse,
    count_parameters,
    simulate_mimo_ber,
)
from models.neural_network import ChannelEstimator
from utils.channel import mimo_rayleigh_channel


def test_mse_and_nmse_exactness():
    h_true = np.array([[[1.0 + 0.0j, 0.0 + 1.0j]]], dtype=np.complex128)
    h_est = np.array([[[1.0 + 0.0j, 0.0 + 0.0j]]], dtype=np.complex128)
    mse = compute_mse(h_true, h_est)
    nmse_lin, nmse_db = compute_nmse(h_true, h_est)
    # Error = 1 on 1 of 2 entries -> MSE = 0.5, ||H||^2 = 2 -> NMSE = 0.5 (-3.0103 dB)
    assert np.isclose(mse, 0.5)
    assert np.isclose(nmse_lin, 0.5)
    assert np.isclose(nmse_db, 10.0 * np.log10(0.5))


def test_count_parameters_matches_expected_dnn():
    model = ChannelEstimator(input_dim=256, output_dim=256, hidden_dims=(512, 256, 256), residual_skip=False)
    expected = (256 * 512 + 512) + (512 * 256 + 256) + (256 * 256 + 256) + (256 * 256 + 256)
    assert count_parameters(model) == expected

    model_res = ChannelEstimator(input_dim=256, output_dim=256, hidden_dims=(512, 256, 256), residual_skip=True)
    assert count_parameters(model_res) == expected + (256 * 256 + 256)


def test_ber_decreases_with_snr_and_improves_with_better_csi():
    rng = np.random.default_rng(77)
    H = mimo_rayleigh_channel(500, nr=4, nt=16, rng=rng)
    # Perfect CSI vs noisy CSI
    H_noisy = H + 0.4 * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape)) / np.sqrt(2.0)

    ber_low_snr = simulate_mimo_ber(H, H, snr_db=0.0, num_data_symbols=40, seed=1)
    ber_high_snr = simulate_mimo_ber(H, H, snr_db=15.0, num_data_symbols=40, seed=1)
    ber_noisy_csi = simulate_mimo_ber(H, H_noisy, snr_db=15.0, num_data_symbols=40, seed=1)

    assert ber_high_snr < ber_low_snr
    assert ber_high_snr <= ber_noisy_csi
