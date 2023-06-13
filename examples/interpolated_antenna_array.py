"""
Interpolated Antenna array gain
================================
"""
import time
import matplotlib.pyplot as plt

import pyant

beam = pyant.beam_of_radar("e3d_stage1", "array")
print(beam)

interp_beam = pyant.models.InterpolatedArray()

interp_beam.generate_interpolation(beam, resolution=(200, 200, None))

fig, axes = plt.subplots(2, 2)

start_time = time.time()
pyant.plotting.gain_heatmap(beam, ax=axes[0, 0], resolution=100, min_elevation=80.0)
axes[0, 0].set_title("Array az=0, el=90")
array_time = time.time() - start_time

start_time = time.time()
pyant.plotting.gain_heatmap(interp_beam, ax=axes[1, 0], resolution=100, min_elevation=80.0)
axes[1, 0].set_title("Interpolated az=0, el=90")
interp_time = time.time() - start_time

# pointing causes no slow-down
interp_beam.sph_point(elevation=30.0, azimuth=45.0)
beam.sph_point(elevation=30.0, azimuth=45.0)

pyant.plotting.gain_heatmap(beam, ax=axes[0, 1], resolution=100, min_elevation=80.0)
pyant.plotting.gain_heatmap(interp_beam, ax=axes[1, 1], resolution=100, min_elevation=80.0)
axes[0, 1].set_title("Array az=30, el=45")
axes[1, 1].set_title("Interpolated az=30, el=45")

print(f"Heatmap plot antenna array: {array_time:.1f} seconds")
print(f"Heatmap plot interpolated array: {interp_time:.1f} seconds")
print(f"Speedup = factor of {array_time/interp_time:.2f}")

plt.show()
