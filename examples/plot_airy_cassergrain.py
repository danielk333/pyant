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

ant_c = pyant.Cassegrain(
    azimuth=0,
    elevation=90.0, 
    frequency=930e6,
    I0=10**4.81,
    a0=23.0,
    a1=40.0,
)

pyant.plotting.gain_heatmap(ant, resolution=300, min_elevation=80.0)
pyant.plotting.gain_heatmap(ant_c, resolution=300, min_elevation=80.0)
pyant.plotting.show()