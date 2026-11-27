"""Unit tests for 6G XL-MIMO near-field spherical-wave and far-field planar-wave channel models."""

import numpy as np

from config.config import DEFAULT_XL_MIMO
from utils.channel import ula_element_positions, xl_mimo_channel_batch


def test_ula_positions_and_rayleigh_distance():
    cfg = DEFAULT_XL_MIMO
    coords = ula_element_positions(cfg.nt, cfg.antenna_spacing_m)
    assert coords.shape == (cfg.nt, 3)
    aperture = float(coords[-1, 0] - coords[0, 0])
    assert np.isclose(aperture, cfg.tx_aperture_m, rtol=1e-6)
    assert cfg.rayleigh_distance_m > max(cfg.near_field_distance_range_m)


def test_xl_mimo_near_vs_far_field_energy_and_spatial_nonstationarity():
    cfg = DEFAULT_XL_MIMO
    rng = np.random.default_rng(42)
    h_nf, meta_nf = xl_mimo_channel_batch(
        num_samples=64,
        nr=cfg.nr,
        nt=cfg.nt,
        wavelength_m=cfg.wavelength_m,
        spacing_m=cfg.antenna_spacing_m,
        distance_range_m=cfg.near_field_distance_range_m,
        non_stationary_visibility=True,
        visibility_fraction_range=(0.5, 0.5),
        regime="near_field",
        rng=rng,
    )
    h_ff, meta_ff = xl_mimo_channel_batch(
        num_samples=64,
        nr=cfg.nr,
        nt=cfg.nt,
        wavelength_m=cfg.wavelength_m,
        spacing_m=cfg.antenna_spacing_m,
        distance_range_m=(cfg.far_field_distance_m, cfg.far_field_distance_m),
        non_stationary_visibility=False,
        regime="far_field",
        rng=rng,
    )
    assert h_nf.shape == (64, cfg.nr, cfg.nt)
    assert h_ff.shape == (64, cfg.nr, cfg.nt)
    # Exactly half of TX elements should be zeroed due to visibility region (non-stationarity)
    zero_cols = np.sum(np.all(np.abs(h_nf) == 0.0, axis=1), axis=1)
    assert np.all(zero_cols == cfg.nt // 2)
    assert np.all(meta_ff["visibility_masks"] == 1.0)
    # Unit average per-entry power preserved
    assert 0.85 < float(np.mean(np.abs(h_nf) ** 2)) < 1.15
    assert 0.85 < float(np.mean(np.abs(h_ff) ** 2)) < 1.15
