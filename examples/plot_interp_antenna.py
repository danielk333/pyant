'''
Interpolated Antenna array gain
================================
'''
import time

import numpy as np
import matplotlib.pyplot as plt

import pyant

xv, yv = np.meshgrid(np.linspace(-50, 50, num=22),
                     np.linspace(-50, 50, num=22))
antennas = np.zeros((3, 22**2))
antennas[0, :] = xv.flatten()
antennas[1, :] = yv.flatten()

beam = pyant.Array(
    azimuth=0,
    elevation=90.0, 
    frequency=46.5e6,
    antennas=antennas,
)

interp_beam = pyant.InterpolatedArray(
    azimuth=0,
    elevation=90.0, 
    frequency=46.5e6,
)

interp_beam.generate_interpolation(beam, resolution=150)

fig, axes = plt.subplots(2, 2)

start_time = time.time()
pyant.plotting.gain_heatmap(beam, ax=axes[0, 0], resolution=100, min_elevation=80.0)
axes[0, 0].set_title('Array az=0, el=90')
array_time = time.time() - start_time

start_time = time.time()
pyant.plotting.gain_heatmap(interp_beam, ax=axes[1, 0], resolution=100, min_elevation=80.0)
axes[1, 0].set_title('Interpolated az=0, el=90')
interp_time = time.time() - start_time

# pointing causes no slow-down
interp_beam.sph_point(elevation=30.0, azimuth=45.0)
beam.sph_point(elevation=30.0, azimuth=45.0)

pyant.plotting.gain_heatmap(beam, ax=axes[0, 1], resolution=100, min_elevation=80.0)
pyant.plotting.gain_heatmap(interp_beam, ax=axes[1, 1], resolution=100, min_elevation=80.0)
axes[0, 1].set_title('Array az=30, el=45')
axes[1, 1].set_title('Interpolated az=30, el=45')

print(f'Heatmap plot antenna array: {array_time:.1f} seconds')
print(f'Heatmap plot interpolated array: {interp_time:.1f} seconds')
print(f'Speedup = factor of {array_time/interp_time:.2f}')

pyant.plotting.show()
