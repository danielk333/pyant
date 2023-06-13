"""
Interpolated arbitrary beam
================================

For use when models need to be evaluated often but the model computation is slow
"""
import time

import matplotlib.pyplot as plt

import pyant

beam = pyant.models.Gaussian(
    azimuth=0,
    elevation=90.0,
    frequency=46.5e6,
    I0=10**4.81,
    radius=100.0,
    normal_azimuth=0,
    normal_elevation=90.0,
    degrees=True,
)

interp_beam = pyant.models.Interpolated(
    azimuth=0,
    elevation=90.0,
    frequency=46.5e6,
    degrees=True,
)

interp_beam.generate_interpolation(beam, resolution=150)

fig, axes = plt.subplots(1, 2)

start_time = time.time()
pyant.plotting.gain_heatmap(beam, ax=axes[0], resolution=1000, min_elevation=80.0)
axes[0].set_title("Gaussian az=0, el=90")
array_time = time.time() - start_time

start_time = time.time()
pyant.plotting.gain_heatmap(interp_beam, ax=axes[1], resolution=1000, min_elevation=80.0)
axes[1].set_title("Interpolated az=0, el=90")
interp_time = time.time() - start_time

print(f"Heatmap plot Gaussian array: {array_time:.1f} seconds")
print(f"Heatmap plot interpolated array: {interp_time:.1f} seconds")

plt.show()
