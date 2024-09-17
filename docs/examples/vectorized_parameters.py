# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Vectorized beams

import time
import pyant
import numpy as np


k_num = 100
f_num = 100
size = (k_num, f_num)
num = np.prod(size)

ks = np.zeros((3, k_num), dtype=np.float64)
ks[0, :] = np.linspace(-0.5, 0.5, num=k_num)
ks[2, :] = np.sqrt(1 - ks[0, :] ** 2 - ks[1, :] ** 2)
fs = np.linspace(200e6, 930e6, num=f_num)


start_time = time.time()

beam = pyant.models.Airy(
    azimuth=45,
    elevation=75.0,
    frequency=930e6,
    I0=10**4.81,
    radius=23.0,
)
g = np.empty(size, dtype=np.float64)
for k_ind in range(k_num):
    for f_ind in range(f_num):
        beam.frequency = fs[f_ind]
        g[k_ind, f_ind] = beam.gain(ks[:, k_ind])

print(f"g.shape={g.shape}")
ex_time_loop = time.time() - start_time
print(f'"gain calculations" ({num}) loop       performance: {ex_time_loop:.1e} seconds')

start_time = time.time()

g = np.empty(size, dtype=np.float64)
for f_ind in range(f_num):
    beam.frequency = fs[f_ind]
    g[:, f_ind] = beam.gain(ks).flatten()

print(f"g.shape={g.shape}")
ex_time_half_loop = time.time() - start_time
print(f'"gain calculations" ({num}) param-loop performance: {ex_time_half_loop:.1e} seconds')


start_time = time.time()

beam.frequency = fs
g = beam.gain(ks)
print(f"g.shape={g.shape}")
ex_time_vectorized = time.time() - start_time
print(f'"gain calculations" ({num}) vectorized performance: {ex_time_vectorized:.1e} seconds')

print(f"loop           vs parameter-loop speedup = {ex_time_loop/ex_time_half_loop:.2f}")
print(f"parameter-loop vs vectorized     speedup = {ex_time_half_loop/ex_time_vectorized:.2f}")
print(f"loop           vs vectorized     speedup = {ex_time_loop/ex_time_vectorized:.2f}")
