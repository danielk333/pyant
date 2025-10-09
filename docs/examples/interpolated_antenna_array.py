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

# # Interpolated Antenna array gain

import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import pyant

# +
np.random.seed(2934)

subgroup_ant_num = 4 * 10 + 1
subgroups = 4
antennas = np.zeros((3, subgroup_ant_num, subgroups))

frequency = 50e6
wavelength = scipy.constants.c / frequency

subgroup = np.zeros((3, subgroup_ant_num))
r = 0.5
for ind in range(1, subgroup_ant_num):
    subgroup[ind % 2, ind] = r * wavelength
    if ind % 2 == 0 and ind != 0:
        r = -r
    if ind % 4 == 0 and ind != 0:
        r += np.sign(r) * 0.5
max_r = np.abs(r)

for ind in range(subgroups):
    phi = np.random.rand() * 2 * np.pi
    antennas[0, :, ind] = subgroup[0, :] + max_r * 3 * np.cos(phi)
    antennas[1, :, ind] = subgroup[1, :] + max_r * 3 * np.sin(phi)

beam = pyant.models.Array(
    pointing=np.array([0, 0, 1], dtype=np.float64),
    frequency=frequency,
    antennas=antennas,
)

# +

fig, ax = plt.subplots()
pyant.plotting.antenna_configuration(beam.antennas, ax=ax)
ax.axis("equal")

# +

interp_beam = pyant.models.InterpolatedArray()
start_time = time.time()
interp_beam.generate_interpolation(
    beam, resolution=(400, 400, None), min_elevation=60.0, interpolate_channels=[0]
)
generate_time = time.time() - start_time
print(f"Interpolation generate: {generate_time:.1e} seconds")

# +
k = pyant.coordinates.sph_to_cart(np.array([20.0, 75.0, 1.0]), degrees=True)

subarray_gains = interp_beam.channel_gain(k)
array_gains = interp_beam.gain(k)
print(f"{k=}")
print(f"{subarray_gains[0]=}")
print(f"{array_gains=}")

# +
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

start_time = time.time()
pyant.plotting.gain_heatmap(beam, ax=axes[0], resolution=400, min_elevation=70.0)
axes[0].set_title("Array")
array_time = time.time() - start_time

start_time = time.time()
pyant.plotting.gain_heatmap(interp_beam, ax=axes[1], resolution=400, min_elevation=70.0)
axes[1].set_title("Interpolated")
interp_time = time.time() - start_time

print(f"Heatmap plot antenna array: {array_time:.1e} seconds")
print(f"Heatmap plot interpolated array: {interp_time:.1e} seconds")
print(f"Speedup = factor of {array_time/interp_time:.2f}")


plt.show()
