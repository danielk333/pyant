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


num = 500
fs = np.linspace(200e6, 930e6, num=num)

beam = pyant.models.Airy(
    pointing=pyant.coordinates.sph_to_cart(np.array([0, 80.0, 1]), degrees=True),
    frequency=930e6,
    radius=23.0,
    peak_gain=10**4.81,
)
beam_vector_params = pyant.models.Airy(
    pointing=np.broadcast_to(
        pyant.coordinates.sph_to_cart(np.array([0, 80.0, 1]), degrees=True).reshape(3, 1), (3, num)
    ),
    frequency=fs,
    radius=np.full((num,), 23.0, dtype=np.float64),
    peak_gain=10**4.81,
)

sph = np.stack(
    [
        np.full((num,), 0, dtype=np.float64),
        np.linspace(90.0, 45, num=num),
        np.full((num,), 1, dtype=np.float64),
    ]
)
k = pyant.coordinates.sph_to_cart(sph, degrees=True)

start_time = time.time()
g = np.empty(num, dtype=np.float64)
for ind in range(num):
    beam.parameters["frequency"] = fs[ind]
    g[ind] = beam.gain(k[:, ind])
ex_time_loop = time.time() - start_time
print(f'"gain calculations" ({num}) loop       performance: {ex_time_loop:.1e} seconds')

start_time = time.time()
g = beam_vector_params.gain(k)
ex_time_vectorized = time.time() - start_time
print(f'"gain calculations" ({num}) param-loop performance: {ex_time_vectorized:.1e} seconds')

# If we want to check only one direction, its the same time as the loop case
# however it can also be computed with the vectorized version

start_time = time.time()
g = beam_vector_params.gain(k[:, 0])
print(f"g.shape={g.shape}")
ex_time_single_vectorized = time.time() - start_time
print(
    f'"gain calculations" ({num}) vectorized performance: {ex_time_single_vectorized:.1e} seconds'
)

print(f"loop set    vs vectorized speedup = {ex_time_loop/ex_time_vectorized:.2f}")
print(f"loop single vs vectorized speedup = {ex_time_loop/ex_time_single_vectorized:.2f}")
