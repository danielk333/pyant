"""
Surface plot
=============
"""
import matplotlib.pyplot as plt
import pyant

beam = pyant.models.Cassegrain(
    azimuth=0,
    elevation=90.0,
    frequency=930e6,
    I0=10**4.81,
    outer_radius=40.0,
    inner_radius=23.0,
    degrees=True,
)

pyant.plotting.gain_surface(beam, resolution=301, min_elevation=85.0)

plt.show()
