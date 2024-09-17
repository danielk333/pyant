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
import matplotlib.pyplot as plt
import pyant


beam = pyant.beam_of_radar("e3d_stage1", "array")
interp_beam = pyant.models.InterpolatedArray()
start_time = time.time()
interp_beam.generate_interpolation(beam, resolution=(400, 400, None), min_elevation=60.0)
generate_time = time.time() - start_time


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

start_time = time.time()
pyant.plotting.gain_heatmap(beam, ax=axes[0], resolution=100, min_elevation=80.0)
axes[0].set_title("Array")
array_time = time.time() - start_time

start_time = time.time()
pyant.plotting.gain_heatmap(interp_beam, ax=axes[1], resolution=100, min_elevation=80.0)
axes[1].set_title("Interpolated")
interp_time = time.time() - start_time

print(f"Interpolation generate: {generate_time:.1f} seconds")
print(f"Heatmap plot antenna array: {array_time:.1f} seconds")
print(f"Heatmap plot interpolated array: {interp_time:.1f} seconds")
print(f"Speedup = factor of {array_time/interp_time:.2f}")

plt.show()
