'''
Random arrays
===========================
'''
import numpy as np

import pyant

np.random.seed(1234)

ant_n = 300

antennas = np.zeros((3, ant_n))
antennas[:2,:] = np.random.rand(2,ant_n)*50-25

beam = pyant.Array(
    azimuth=0,
    elevation=90.0, 
    frequency=46.5e6,
    antennas=antennas,
)
pyant.plotting.gain_heatmap(beam, resolution=100, min_elevation=60.0)
pyant.plotting.show()
