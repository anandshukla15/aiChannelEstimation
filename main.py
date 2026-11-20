import numpy as np

from utils.modulation import generate_bits
from utils.modulation import bpsk_modulate
from utils.channel import rayleigh_channel
from utils.noise import add_awgn


NUM_BITS = 1000

SNR = 15


bits = generate_bits(NUM_BITS)

symbols = bpsk_modulate(bits)

H = rayleigh_channel(NUM_BITS)

received = H * symbols

received = add_awgn(received, SNR)

print("Bits:")
print(bits[:10])

print()

print("BPSK Symbols:")
print(symbols[:10])

print()

print("Channel:")
print(H[:5])

print()

print("Received Signal:")
print(received[:5])