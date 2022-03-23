'''
Collection of beams
===================

This assumes that the raw voltage data from the beams are analyzed in such a 
way that the complex voltages have the same direction in complex space upon 
summation. 
'''

import functools
import operator

import pyant
import numpy as np
import matplotlib.pyplot as plt


beams = [
    pyant.FiniteCylindricalParabola(
        azimuth=0,
        elevation=el, 
        frequency=224.0e6,
        I0=10**4.81,
        width=30.0,
        height=40.0,
    )
    for el in [90.0, 80.0, 70.0, 60.0]
]

k = np.array([0, 0, 1])

gains = [b.gain(k) for b in beams]
print(f'Individual gains {np.log10(gains)*10} dB')

gain_sum = functools.reduce(operator.add, gains)
print(f'Summed gains {np.log10(gain_sum)*10} dB')

print(f'Gain of beam 2 {beams[1].gain(k)}')

fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=80)
for beam, ax in zip(beams, axes.flatten()):
    pyant.plotting.gain_heatmap(beam, ax=ax)

pyant.plotting.gain_heatmap(beams)

pyant.plotting.show()
