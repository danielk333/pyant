'''
Phase steerable Half-pipe
============================
'''

import numpy as np
import matplotlib.pyplot as plt

import pyant


beam = pyant.PhasedFiniteCylindricalParabola(
    azimuth=0,
    elevation=90.0, 
    depth=18,
    phase_steering = [5.0,10.0,25.0,30.0],
    frequency=224.0e6,
    I0=10**4.81,
    width=120.0,
    height=40.0,
)

fig, axes = plt.subplots(2,2,figsize=(10,6),dpi=80)
axes = axes.flatten()
for i in range(beam.named_shape()['phase_steering']):
    pyant.plotting.gain_heatmap(
        beam, 
        resolution=901, 
        min_elevation=30.0, 
        ax=axes[i],
        ind = {
            "phase_steering":i,
        },
    )
    axes[i].set_title(f'{int(beam.phase_steering[i])} deg steering')

pyant.plotting.show()