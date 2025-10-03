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

# # Vectorized gain functions

import time
import numpy as np
import pyant


kn = 500
beam = pyant.models.Airy(
    pointing=pyant.coordinates.sph_to_cart(np.array([0, 80.0, 1]), degrees=True),
    frequency=930e6,
    peak_gain=10**4.81,
    radius=23.0,
)

sph = np.stack(
    [
        np.full((kn,), 0, dtype=np.float64),
        np.linspace(90.0, 45, num=kn),
        np.full((kn,), 1, dtype=np.float64),
    ]
)
k = pyant.coordinates.sph_to_cart(sph, degrees=True)


# loop version
start_time = time.time()
gain = np.full((kn,), np.nan, dtype=np.float64)
for ind in range(kn):
    gain[ind] = beam.gain(k[:, ind])
loop_time = time.time() - start_time

# vectorized version
start_time = time.time()
gain = beam.gain(k)
vector_time = time.time() - start_time

print(f'"Airy.gain" ({kn}) loop       performance: {loop_time:.1e} seconds')
print(f'"Airy.gain" ({kn}) vectorized performance: {vector_time:.1e} seconds')
print(f"Speedup = {loop_time/vector_time}")


# Can also do vectorized spherical arguments, there is some extra overhead for this feature
start_time = time.time()
gain_sph = beam.sph_gain(azimuth=np.linspace(0, 360, num=kn), elevation=np.ones((kn,)) * 75.0)
sph_vector_time = time.time() - start_time


print(f'"Airy.sph_gain" ({kn}) vectorized performance: {sph_vector_time:.1e} seconds')
print(f"Speedup = {vector_time/sph_vector_time}")
