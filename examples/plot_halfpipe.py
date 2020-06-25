'''
Halfpipe radar
===========================
'''

import pyant

beam = pyant.FiniteCylindricalParabola(
    azimuth=0,
    elevation=90.0, 
    frequency=224.0e6,
    I0=10**4.81,
    width=120.0,
    height=40.0,
)


pyant.plotting.gain_heatmap(beam, resolution=300, min_elevation=80.0)
pyant.plotting.show()