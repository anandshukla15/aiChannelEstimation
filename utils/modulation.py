import numpy as np


def generate_bits(num_bits):
    """
    Generate random binary bits.
    """
    return np.random.randint(0, 2, num_bits)


def bpsk_modulate(bits):
    """
    BPSK Modulation

    0 -> -1
    1 -> +1
    """

    return 2 * bits - 1