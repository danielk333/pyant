'''
Interpolated Antenna array gain
================================
'''
import time

import numpy as np

import pyant

xv, yv = np.meshgrid(np.linspace(-50,50, num=22), np.linspace(-50,50, num=22))
antennas = np.zeros((3,22**2))
antennas[0,:] = xv.flatten()
antennas[1,:] = yv.flatten()

beam = pyant.Array(
    azimuth=0,
    elevation=90.0, 
    frequency=46.5e6,
    antennas=antennas,
)

interp_beam = pyant.PlaneArrayInterp(
    azimuth=0,
    elevation=90.0, 
    frequency=46.5e6,
)

interp_beam.generate_interpolation(beam, resolution=150)

start_time = time.time()
pyant.plotting.gain_heatmap(beam, resolution=100, min_elevation=80.0)
array_time = time.time() - start_time

start_time = time.time()
pyant.plotting.gain_heatmap(interp_beam, resolution=100, min_elevation=80.0)
interp_time = time.time() - start_time

#can also copy interpolations
interp_beam2 = interp_beam.copy()

#pointing causes no slow-down
interp_beam.sph_point(elevation=30.0, azimuth=45.0)
beam.sph_point(elevation=30.0, azimuth=45.0)

interp_beam2.sph_point(elevation=45.0, azimuth=120.0)

pyant.plotting.gain_heatmap(interp_beam2, resolution=100, min_elevation=80.0)
pyant.plotting.gain_heatmap(interp_beam, resolution=100, min_elevation=80.0)
pyant.plotting.gain_heatmap(beam, resolution=100, min_elevation=80.0)

print(f'Heatmap plot antenna array: {array_time:.1f} seconds')
print(f'Heatmap plot interpolated array: {interp_time:.1f} seconds')
print(f'Speedup = factor of {array_time/interp_time:.2f}')

pyant.plotting.show()