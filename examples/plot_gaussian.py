'''
Gaussian planar array gain
===========================
'''

import pyant

beam = pyant.models.Gaussian(
    azimuth=0,
    elevation=90.0,
    frequency=46.5e6,
    I0=10**4.81,
    radius=100.0,
    normal_azimuth = 0,
    normal_elevation = 90.0,
)


pyant.plotting.gain_heatmap(beam, resolution=300, min_elevation=80.0)
pyant.plotting.show()
