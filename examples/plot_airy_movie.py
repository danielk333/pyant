'''
Airy disk movie
========================
'''
import numpy as np
import pyant

beam = pyant.Airy(
    azimuth=0,
    elevation=90.0, 
    frequency=50e6,
    I0=10**4.81,
    radius=10.0,
)

def update(beam, item):
    beam.radius = item
    beam.elevation += 0.25
    return beam

pyant.plotting.gain_heatmap_movie(beam, iterable=np.linspace(10,23,num=100), beam_update=update)
pyant.plotting.show()