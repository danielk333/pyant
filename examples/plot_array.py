'''
Antenna array gain
===========================
'''
import numpy as np

import pyant

xv, yv = np.meshgrid(np.linspace(-50,50, num=22), np.linspace(-50,50, num=22))
antennas = np.zeros((22**2, 3))
antennas[:,0] = xv.flatten()
antennas[:,1] = yv.flatten()

ant = pyant.Array(
    azimuth=0,
    elevation=90.0, 
    frequency=46.5e6,
    antennas=antennas,
)


pyant.plotting.gain_heatmap(ant, resolution=100, min_elevation=80.0)
pyant.plotting.show()