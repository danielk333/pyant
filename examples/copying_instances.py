'''
Copy of beams
========================
'''

import pyant

import numpy as np

beam = pyant.Airy(
    azimuth=0,
    elevation=90.0, 
    frequency=[930e6, 220e6],
    I0=10**4.81,
    radius=23.0,
)

beam_2 = beam.copy()

beam_2.sph_point(
    azimuth=0,
    elevation=45.0, 
)
beam_2.frequency = beam_2.frequency[1:]

pyant.plotting.gain_heatmap(beam, resolution=301, min_elevation=80.0)
pyant.plotting.gain_heatmap(beam_2, resolution=301, min_elevation=80.0)
pyant.plotting.show()

xv, yv = np.meshgrid(np.linspace(-50, 50, num=22),
                     np.linspace(-50, 50, num=22))
antennas = np.zeros((3, 22**2))
antennas[0, :] = xv.flatten()
antennas[1, :] = yv.flatten()

arr = pyant.Array(
    azimuth=0,
    elevation=90.0, 
    frequency=46.5e6,
    antennas=antennas,
)

arr2 = arr.copy()
arr2.antennas[:100, 1] += 25
arr2.antennas[:100, 0] -= 25

arr.sph_point(
    azimuth=0,
    elevation=80, 
)

pyant.plotting.gain_heatmap(arr, resolution=301, min_elevation=80.0)
pyant.plotting.gain_heatmap(arr2, resolution=301, min_elevation=80.0)
pyant.plotting.show()
