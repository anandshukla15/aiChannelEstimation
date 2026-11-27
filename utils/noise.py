"""Complex AWGN generation and SNR-to-noise-variance utilities."""

from typing import Optional, Tuple, Union
import numpy as np


def compute_noise_variance(
    snr_db: Union[float, np.ndarray],
    signal_power: Union[float, np.ndarray] = 1.0,
) -> Union[float, np.ndarray]:
    """Compute complex AWGN variance sigma^2 from SNR in dB.

    SNR_linear = signal_power / sigma^2  =>  sigma^2 = signal_power / 10^(snr_db / 10).
    For complex circularly symmetric Gaussian noise N = CN(0, sigma^2),
    Re(N) ~ N(0, sigma^2 / 2) and Im(N) ~ N(0, sigma^2 / 2).
    """
    snr_linear = 10.0 ** (np.asarray(snr_db, dtype=np.float64) / 10.0)
    noise_var = np.asarray(signal_power, dtype=np.float64) / snr_linear
    if np.ndim(snr_db) == 0 and np.ndim(signal_power) == 0:
        return float(noise_var)
    return noise_var


def generate_complex_awgn(
    shape: Tuple[int, ...],
    noise_variance: Union[float, np.ndarray],
    rng: Optional[np.random.Generator] = None,
    dtype: np.dtype = np.complex128,
) -> np.ndarray:
    """Generate complex circularly symmetric AWGN with variance E[|N|^2] = noise_variance.

    N = sqrt(noise_variance / 2) * (N_real + j * N_imag)
    """
    if rng is None:
        real = np.random.randn(*shape)
        imag = np.random.randn(*shape)
    else:
        real = rng.standard_normal(shape)
        imag = rng.standard_normal(shape)

    noise = np.sqrt(np.asarray(noise_variance) / 2.0) * (real + 1j * imag)
    return noise.astype(dtype)


def add_awgn(
    signal: np.ndarray,
    snr_db: Union[float, np.ndarray],
    expected_signal_power: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add complex AWGN to a signal at the specified SNR (dB).

    If `expected_signal_power` is None, empirical mean signal power is used
    (preserving backward compatibility with the original SISO utility).
    For Massive MIMO pilot transmission Y = H @ X + N with E[|H_{i,j}|^2] = 1
    and X @ X^H = Tp * I_{Nt}, expected_signal_power equals Nt so that noise
    variance depends solely on SNR and Nt rather than the channel realization.
    """
    if expected_signal_power is None:
        sig_power = float(np.mean(np.abs(signal) ** 2))
    else:
        sig_power = float(expected_signal_power)

    noise_var = compute_noise_variance(snr_db, signal_power=sig_power)
    noise = generate_complex_awgn(signal.shape, noise_var, rng=rng, dtype=signal.dtype if np.iscomplexobj(signal) else np.complex128)
    return signal + noise
