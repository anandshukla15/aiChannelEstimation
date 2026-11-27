"""Unit tests verifying configurable antenna dimensions across config and estimators."""

import numpy as np
import pytest

from config.config import MIMOConfig
from data.dataset import generate_mimo_dataset
from traditional.ls_estimator import mimo_ls_estimate
from traditional.mmse_estimator import mimo_mmse_estimate


@pytest.mark.parametrize("nt", [8, 16, 32, 64])
@pytest.mark.parametrize("nr", [2, 4, 8])
def test_configurable_antenna_dimensions(nt, nr):
    cfg = MIMOConfig(nt=nt, nr=nr, tp=nt, num_train_val_samples=20)
    assert cfg.input_dim == 2 * nr * nt
    assert cfg.output_dim == 2 * nr * nt

    ds = generate_mimo_dataset(num_samples=16, nt=nt, nr=nr, tp=nt, seed=10 + nt + nr)
    assert ds["H"].shape == (16, nr, nt)
    assert ds["Y"].shape == (16, nr, nt)
    assert ds["inputs"].shape == (16, cfg.input_dim)
    assert ds["targets"].shape == (16, cfg.output_dim)

    h_ls = mimo_ls_estimate(ds["Y"], ds["X"])
    h_mmse = mimo_mmse_estimate(ds["Y"], ds["X"], ds["noise_variance"])
    assert h_ls.shape == (16, nr, nt)
    assert h_mmse.shape == (16, nr, nt)
