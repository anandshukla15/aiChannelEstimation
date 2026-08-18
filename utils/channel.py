import numpy as np


def rayleigh_channel(num_samples):
    """
    Generate Rayleigh fading channel.

    H = (Real + jImag)/sqrt(2)
    """

    real = np.random.randn(num_samples)
    imag = np.random.randn(num_samples)

    h = (real + 1j * imag) / np.sqrt(2)

    return h