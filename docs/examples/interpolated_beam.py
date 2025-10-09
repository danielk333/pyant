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

# # Interpolated arbitrary beam
#
# For use when models need to be evaluated often but the model computation is slow
# Here the Gaussian model is one of the fastest available and the speed is comparable
# with the interpolated beam.

import time
import numpy as np
import matplotlib.pyplot as plt
import pyant

beam = pyant.models.Gaussian(
    pointing=np.array([0, 0, 1], dtype=np.float64),
    frequency=46.5e6,
    radius=100.0,
    normal_pointing=np.array([0, 0, 1], dtype=np.float64),
    peak_gain=10**4.81,
)
interp_beam = pyant.models.Interpolated()
interp_beam.generate_interpolation(beam, resolution=150)


# +
fig, axes = plt.subplots(2, 2, figsize=(12, 5))
axes = axes.flatten()

start_time = time.time()
pyant.plotting.gain_heatmap(beam, ax=axes[0], resolution=1000, min_elevation=80.0)
axes[0].set_title("Gaussian beam")
gauss_time = time.time() - start_time

start_time = time.time()
pyant.plotting.gain_heatmap(interp_beam, ax=axes[1], resolution=1000, min_elevation=80.0)
axes[1].set_title("Interpolated beam 150 resolution - linear")
interp_time = time.time() - start_time

interp_beam.generate_interpolation(beam, resolution=150, interpolation_method="bivariate_spline")

start_time = time.time()
pyant.plotting.gain_heatmap(interp_beam, ax=axes[2], resolution=1000, min_elevation=80.0)
axes[2].set_title("Interpolated beam 150 resolution - spline")
interp_spline_time = time.time() - start_time

interp_beam.generate_interpolation(beam, resolution=500)

start_time = time.time()
pyant.plotting.gain_heatmap(interp_beam, ax=axes[3], resolution=1000, min_elevation=80.0)
axes[3].set_title("Interpolated beam 500 resolution - linear")
interp_high_time = time.time() - start_time

print(f"Heatmap plot Gaussian: {gauss_time:.1f} seconds")
print(f"Heatmap plot interpolated: {interp_time:.1f} seconds")
print(f"Heatmap plot interpolated spline: {interp_spline_time:.1f} seconds")
print(f"Heatmap plot interpolated high-res: {interp_high_time:.1f} seconds")
plt.show()
