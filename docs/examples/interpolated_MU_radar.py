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

raise NotImplementedError()
# # Interpolated MU radar
#
# Combination of interpolated base antenna field and a measured antenna gain pattern
# compared to doing the direct array calculation

import time
import matplotlib.pyplot as plt
import pyant

# +
beam = pyant.beam_of_radar(
    "mu", "interpolated_array", resolution=(500, 500, None), min_elevation=80.0
)
base_beam = pyant.beam_of_radar("mu", "array")

# +
fig, axes = plt.subplots(1, 2)

start_time = time.time()
pyant.plotting.gain_heatmap(
    base_beam,
    resolution=300,
    min_elevation=80.0,
    centered=False,
    ax=axes[0],
)
array_time = time.time() - start_time

start_time = time.time()
axes[0].set_title("Array")
pyant.plotting.gain_heatmap(
    beam,
    resolution=300,
    min_elevation=80.0,
    centered=False,
    ax=axes[1],
)
interp_time = time.time() - start_time
axes[1].set_title("Interpolated array")

print(f"Heatmap plot Array: {array_time:.1f} seconds")
print(f"Heatmap plot InterpolatedArray: {interp_time:.1f} seconds")
print(f"Speedup = factor of {array_time/interp_time:.2f}")

plt.show()
