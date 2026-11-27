"""Wireless channel models: SISO Rayleigh, Massive MIMO Rayleigh, and 6G XL-MIMO near-field."""

from typing import Optional, Tuple
import numpy as np


def rayleigh_channel(
    num_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate 1D SISO complex Rayleigh fading coefficients.

    H = (Real + j * Imag) / sqrt(2), with unit average power E[|H|^2] = 1.
    Preserved for backward compatibility with existing SISO scripts.
    """
    if rng is None:
        real = np.random.randn(num_samples)
        imag = np.random.randn(num_samples)
    else:
        real = rng.standard_normal(num_samples)
        imag = rng.standard_normal(num_samples)

    return (real + 1j * imag) / np.sqrt(2.0)


def mimo_rayleigh_channel(
    num_samples: int,
    nr: int,
    nt: int,
    rng: Optional[np.random.Generator] = None,
    dtype: np.dtype = np.complex128,
) -> np.ndarray:
    """Generate independent complex i.i.d. Rayleigh fading MIMO channel matrices.

    H = (H_r + j * H_i) / sqrt(2), shape (num_samples, nr, nt),
    where each entry H_{i,j} ~ CN(0, 1) independently.
    """
    if rng is None:
        rng = np.random.default_rng()

    real = rng.standard_normal((num_samples, nr, nt))
    imag = rng.standard_normal((num_samples, nr, nt))
    h = (real + 1j * imag) / np.sqrt(2.0)
    return h.astype(dtype)


def ula_element_positions(
    num_elements: int,
    spacing_m: float,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    axis: str = "x",
) -> np.ndarray:
    """Compute 3D coordinates of a centered Uniform Linear Array (ULA).

    Returns array of shape (num_elements, 3) in meters.
    """
    indices = np.arange(num_elements, dtype=np.float64) - (num_elements - 1) / 2.0
    offsets = indices * spacing_m
    coords = np.tile(np.asarray(center, dtype=np.float64), (num_elements, 1))
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    coords[:, axis_idx] += offsets
    return coords


def xl_mimo_channel_batch(
    num_samples: int,
    nr: int,
    nt: int,
    wavelength_m: float,
    spacing_m: float,
    distance_range_m: Tuple[float, float],
    angle_range_deg: Tuple[float, float] = (-60.0, 60.0),
    non_stationary_visibility: bool = True,
    visibility_fraction_range: Tuple[float, float] = (0.4, 1.0),
    rician_k_factor_db: float = 10.0,
    regime: str = "near_field",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, dict]:
    """Generate physically consistent XL-MIMO channels comparing near-field vs far-field.

    Geometry:
        - Transmit ULA centered at origin along the x-axis: positions p_tx[n] = (x_n, 0, 0).
        - Receive ULA of `nr` elements spaced by `spacing_m` centered at distance `r`
          and azimuth `theta` in the x-y plane (parallel to x-axis for clarity).
        - Wavelength lambda = c / f_c.

    Near-field spherical-wave LoS model:
        For TX element n and RX element m with exact Euclidean distance d_{m,n} = ||p_rx[m] - p_tx[n]||_2:
            H_LoS[m, n] = sqrt(nt * nr) * (d_center / d_{m,n}) * exp(-j * 2 * pi / lambda * d_{m,n}) / norm_factor
        normalized so that E[||H||_F^2] = nr * nt (unit average per-entry power, allowing
        direct comparison against i.i.d. Rayleigh channels at matched receive SNR).
        Spatial non-stationarity (SNS) in XL-MIMO is modeled via a contiguous Visibility Region (VR)
        mask across the large TX array (each sample sees a contiguous subarray of length
        floor(frac * nt) active elements due to blockage/near-field scatterer visibility).

    Far-field planar-wave LoS model:
        Uses the classic planar-wave steering vectors a_rx(theta), a_tx(theta) with linear
        phase progression (ignoring spherical wavefront curvature and visibility blockage).

    Rician NLOS component:
        Combines LoS and scattered i.i.d. Rayleigh NLOS (within the visible subarray) with
        Rician K-factor `rician_k_factor_db`.

    Returns:
        H: complex128 array of shape (num_samples, nr, nt)
        metadata: dict containing distances, angles, and visibility masks.
    """
    if rng is None:
        rng = np.random.default_rng()

    tx_coords = ula_element_positions(nt, spacing_m, center=(0.0, 0.0, 0.0), axis="x")
    rx_rel = ula_element_positions(nr, spacing_m, center=(0.0, 0.0, 0.0), axis="x")

    k_wave = 2.0 * np.pi / wavelength_m
    k_lin = 10.0 ** (rician_k_factor_db / 10.0)
    los_weight = np.sqrt(k_lin / (k_lin + 1.0))
    nlos_weight = np.sqrt(1.0 / (k_lin + 1.0))

    distances = rng.uniform(distance_range_m[0], distance_range_m[1], size=num_samples)
    angles_deg = rng.uniform(angle_range_deg[0], angle_range_deg[1], size=num_samples)
    angles_rad = np.deg2rad(angles_deg)

    h_all = np.zeros((num_samples, nr, nt), dtype=np.complex128)
    visibility_masks = np.ones((num_samples, nt), dtype=np.float64)

    for idx in range(num_samples):
        r = distances[idx]
        theta = angles_rad[idx]
        rx_center = np.array([r * np.sin(theta), r * np.cos(theta), 0.0], dtype=np.float64)
        rx_coords = rx_rel + rx_center[None, :]

        if regime == "near_field":
            # Exact spherical-wave Euclidean distance between every RX-TX element pair
            diff = rx_coords[:, None, :] - tx_coords[None, :, :]  # (nr, nt, 3)
            d_mn = np.linalg.norm(diff, axis=-1)  # (nr, nt)
            # Normalized spherical wave amplitude & exact phase
            amp = r / np.maximum(d_mn, 1e-6)
            h_los = amp * np.exp(-1j * k_wave * (d_mn - r))

            if non_stationary_visibility:
                frac = rng.uniform(visibility_fraction_range[0], visibility_fraction_range[1])
                vis_len = max(1, min(nt, int(np.round(frac * nt))))
                start = int(rng.integers(0, nt - vis_len + 1))
                mask = np.zeros(nt, dtype=np.float64)
                mask[start : start + vis_len] = 1.0
                # Power normalization across visible elements to preserve E[||H||_F^2] = nr * nt
                scale = np.sqrt(nt / vis_len)
                visibility_masks[idx] = mask
                h_los = h_los * (mask[None, :] * scale)
                nlos_raw = (rng.standard_normal((nr, nt)) + 1j * rng.standard_normal((nr, nt))) / np.sqrt(2.0)
                h_nlos = nlos_raw * (mask[None, :] * scale)
            else:
                h_nlos = (rng.standard_normal((nr, nt)) + 1j * rng.standard_normal((nr, nt))) / np.sqrt(2.0)

        elif regime == "far_field":
            # Planar-wave approximation: linear phase across ULA, uniform amplitude
            tx_idx = np.arange(nt, dtype=np.float64) - (nt - 1) / 2.0
            rx_idx = np.arange(nr, dtype=np.float64) - (nr - 1) / 2.0
            a_tx = np.exp(-1j * k_wave * spacing_m * np.sin(theta) * tx_idx)
            a_rx = np.exp(-1j * k_wave * spacing_m * np.sin(theta) * rx_idx)
            h_los = np.outer(a_rx, a_tx.conj())
            h_nlos = (rng.standard_normal((nr, nt)) + 1j * rng.standard_normal((nr, nt))) / np.sqrt(2.0)
        else:
            raise ValueError(f"Unknown XL-MIMO regime '{regime}'. Use 'near_field' or 'far_field'.")

        # Normalize LoS Frobenius energy to nr * nt before Rician combining so total channel power matches Rayleigh
        los_norm = np.linalg.norm(h_los, ord="fro")
        if los_norm > 0:
            h_los = h_los * (np.sqrt(nr * nt) / los_norm)

        h_all[idx] = los_weight * h_los + nlos_weight * h_nlos

    metadata = {
        "distances_m": distances,
        "angles_deg": angles_deg,
        "visibility_masks": visibility_masks,
        "tx_coords_m": tx_coords,
    }
    return h_all, metadata
