"""
Vectorized gain functions
================================
"""
import time

import numpy as np
import pyant

kn = 500

beam = pyant.models.Airy(
    azimuth=45.0,
    elevation=75.0,
    frequency=930e6,
    I0=10**4.81,
    radius=23.0,
    degrees=True,
)

kx = np.linspace(-1, 1, num=kn)
ky = np.linspace(-1, 1, num=kn)

size = kn**2

xv, yv = np.meshgrid(kx, ky, sparse=False, indexing="ij")
k = np.zeros((3, size), dtype=np.float64)
k[0, :] = xv.reshape(1, size)
k[1, :] = yv.reshape(1, size)
xy2 = k[0, :] ** 2 + k[1, :] ** 2
inds = xy2 <= 1
k[2, inds] = np.sqrt(1.0 - xy2[inds])


# loop version
start_time = time.time()

G = np.full((size,), np.nan, dtype=np.float64)
for i in range(size):
    if inds[i]:
        G[i] = beam.gain(k[:, i])
G = G.reshape(kn, kn)

loop_time = time.time() - start_time

# vectorized version
start_time = time.time()

G = np.full((size,), np.nan, dtype=np.float64)
G[inds] = beam.gain(k[:, inds]).flatten()
G = G.reshape(kn, kn)

vector_time = time.time() - start_time

print(f'"Airy.gain" ({size}) loop       performance: {loop_time:.1e} seconds')
print(f'"Airy.gain" ({size}) vectorized performance: {vector_time:.1e} seconds')
print(f"Speedup = {loop_time/vector_time}")


# Can also do vectorized spherical arguments, there is some extra overhead for this feature
start_time = time.time()
G_sph = beam.sph_gain(azimuth=np.linspace(0, 360, num=size), elevation=np.ones((size,)) * 75.0)
vector_time = time.time() - start_time


start_time = time.time()
for az in np.linspace(0, 360, num=size):
    G_sph = beam.sph_gain(azimuth=az, elevation=75.0)
loop_time = time.time() - start_time

print(f'"Airy.sph_gain" ({size}) loop       performance: {loop_time:.1e} seconds')
print(f'"Airy.sph_gain" ({size}) vectorized performance: {vector_time:.1e} seconds')
print(f"Speedup = {loop_time/vector_time}")
