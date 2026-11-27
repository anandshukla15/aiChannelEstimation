"""Bit generation and digital modulation/demodulation utilities."""

from typing import Optional
import numpy as np


def generate_bits(
    num_bits,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate random binary bits {0, 1}.

    Accepts either an integer count or a tuple shape for MIMO bit streams.
    """
    if rng is None:
        return np.random.randint(0, 2, size=num_bits)
    return rng.integers(0, 2, size=num_bits)


def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """BPSK Modulation: 0 -> -1, 1 -> +1 (unit symbol energy)."""
    return (2 * np.asarray(bits) - 1).astype(np.float64)


def bpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    """Hard-decision BPSK Demodulation: Re(sym) >= 0 -> 1, else 0."""
    return (np.real(symbols) >= 0).astype(np.int64)


def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """Unit-energy Gray-coded QPSK modulation.

    Expects last dimension of `bits` to be even (pairs of bits [b0, b1]).
    """
    bits_arr = np.asarray(bits)
    if bits_arr.shape[-1] % 2 != 0:
        raise ValueError("Last dimension of bits must be even for QPSK modulation.")
    b0 = bits_arr[..., 0::2]
    b1 = bits_arr[..., 1::2]
    i_comp = 2 * b0 - 1
    q_comp = 2 * b1 - 1
    return (i_comp + 1j * q_comp) / np.sqrt(2.0)


def qpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    """Hard-decision QPSK demodulation matching `qpsk_modulate`."""
    sym = np.asarray(symbols)
    b0 = (np.real(sym) >= 0).astype(np.int64)
    b1 = (np.imag(sym) >= 0).astype(np.int64)
    out = np.empty(sym.shape[:-1] + (sym.shape[-1] * 2,), dtype=np.int64)
    out[..., 0::2] = b0
    out[..., 1::2] = b1
    return out
