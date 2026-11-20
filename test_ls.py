import numpy as np

from utils.modulation import generate_bits, bpsk_modulate
from utils.channel import rayleigh_channel
from utils.noise import add_awgn

from traditional.ls_estimator import ls_channel_estimation


# Number of symbols
NUM_PILOTS = 1000

# SNR
SNR = 15


# Generate known pilot bits
pilot_bits = generate_bits(NUM_PILOTS)

# BPSK modulation
pilot_symbols = bpsk_modulate(pilot_bits)


# Generate one Rayleigh fading channel
H_actual = rayleigh_channel(1)[0]


# Pass pilots through channel
received_pilots = H_actual * pilot_symbols


# Add noise
received_pilots = add_awgn(received_pilots, SNR)


# Estimate channel using LS
H_estimated = ls_channel_estimation(
    received_pilots,
    pilot_symbols
)


# Calculate estimation error
mse = np.abs(H_actual - H_estimated) ** 2


print("Actual Channel:", H_actual)
print("Estimated Channel:", H_estimated)

print("MSE:", mse)