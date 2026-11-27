"""Unit tests for Massive MIMO dataset generation, pilot orthogonality, and AWGN variance."""

import numpy as np
import pytest

from data.dataset import complex_to_real_features, generate_mimo_dataset, real_features_to_complex
from utils.pilots import generate_pilots, verify_pilot_orthogonality


def test_pilot_orthogonality_all_supported_configs():
    for nt in (8, 16, 32, 64):
        tp = nt
        pilots = generate_pilots(nt=nt, tp=tp, method="hadamard", require_orthogonal=True)
        assert pilots.shape == (nt, tp)
        is_ortho, err = verify_pilot_orthogonality(pilots)
        assert is_ortho, f"Pilots for Nt={nt}, Tp={tp} not orthogonal, err={err}"


def test_orthogonal_pilots_impossible_when_tp_lt_nt():
    with pytest.raises(ValueError, match="impossible"):
        generate_pilots(nt=32, tp=16, method="hadamard", require_orthogonal=True)


def test_dataset_dimensions_and_statistical_properties():
    ds = generate_mimo_dataset(
        num_samples=1000,
        nt=16,
        nr=4,
        tp=16,
        snr_db_range=(10.0, 10.0),
        seed=123,
    )
    assert ds["H"].shape == (1000, 4, 16)
    assert ds["X"].shape == (16, 16)
    assert ds["Y"].shape == (1000, 4, 16)
    assert ds["snr_db"].shape == (1000,)
    assert ds["noise_variance"].shape == (1000,)
    assert ds["h_ls"].shape == (1000, 4, 16)
    assert ds["inputs"].shape == (1000, 2 * 4 * 16)
    assert ds["targets"].shape == (1000, 2 * 4 * 16)

    # Check Rayleigh variance E[|H_ij|^2] ~ 1.0
    mean_h_pow = float(np.mean(np.abs(ds["H"]) ** 2))
    assert 0.9 < mean_h_pow < 1.1

    # Check AWGN variance E[|Y - H@X|^2] matches noise_variance (= Nt / SNR_linear = 16 / 10 = 1.6)
    noise_emp = ds["Y"] - (ds["H"] @ ds["X"])
    emp_var = float(np.mean(np.abs(noise_emp) ** 2))
    expected_var = float(np.mean(ds["noise_variance"]))
    assert np.isclose(expected_var, 1.6, rtol=1e-4)
    assert np.isclose(emp_var, expected_var, rtol=0.05)


def test_complex_real_roundtrip_preserves_order():
    rng = np.random.default_rng(7)
    arr = rng.standard_normal((12, 4, 8)) + 1j * rng.standard_normal((12, 4, 8))
    feats = complex_to_real_features(arr)
    assert feats.shape == (12, 2 * 4 * 8)
    recon = real_features_to_complex(feats, nr=4, cols=8)
    assert np.allclose(arr, recon, atol=1e-6)


def test_train_test_datasets_are_independent():
    train_ds = generate_mimo_dataset(num_samples=200, nt=8, nr=2, tp=8, seed=42)
    test_ds = generate_mimo_dataset(num_samples=200, nt=8, nr=2, tp=8, seed=10042)
    assert not np.allclose(train_ds["H"], test_ds["H"])
