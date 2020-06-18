'''
Airy disk antenna gain
========================
'''

import pyant

ant = pyant.Airy(
    azimuth=0,
    elevation=90.0, 
    frequency=930e6,
    I0=10**4.81,
    radius=23.0,
)

pyant.plotting.gain_heatmap(ant, resolution=500, min_elevation=80.0)
pyant.plotting.show()