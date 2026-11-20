import numpy as np


def ls_channel_estimation(received_pilots, transmitted_pilots):
    """
    Estimate wireless channel using Least Squares.

    H_est = sum(conj(X) * Y) / sum(abs(X)^2)
    """

    numerator = np.sum(np.conj(transmitted_pilots) * received_pilots)

    denominator = np.sum(np.abs(transmitted_pilots) ** 2)

    if denominator == 0:
        raise ValueError("Pilot symbols cannot all be zero")

    h_estimated = numerator / denominator

    return h_estimated