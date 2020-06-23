'''
Collection of beams
===================

This assumes that the raw voltage data from the beams are analyzed in such a way that the complex voltages have the same direction in complex space upon summation. 
'''

import pyant
import numpy as np
import matplotlib.pyplot as plt

beams = pyant.Beams([
    pyant.FiniteCylindricalParabola(
        azimuth=0,
        elevation=el, 
        frequency=224.0e6,
        I0=10**4.81,
        width=30.0,
        height=40.0,
    )
    for el in [90.0, 80.0, 70.0, 60.0]
])

fig, axes = plt.subplots(2,2,figsize=(10,6),dpi=80)
for beam, ax in zip(beams, axes.flatten()):
    pyant.plotting.gain_heatmap(beam, ax=ax)

pyant.plotting.gain_heatmap(beams)

pyant.plotting.show()