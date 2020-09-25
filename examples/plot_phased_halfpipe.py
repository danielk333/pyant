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
    frequency=30e6,
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
    #axes[i].axis('scaled')
    axes[i].set_title(f'{int(beam.phase_steering[i])} deg steering')

beam_two = pyant.PhasedFiniteCylindricalParabola(
    azimuth=0,
    elevation=[30., 60., 90.], 
    depth=18,
    phase_steering = [0,45.0],
    frequency=224.0e6,
    width=120.0/4,
    height=40.0,
)


fig, axes = plt.subplots(2,3,figsize=(10,6),dpi=80)
for i in range(beam_two.named_shape()['phase_steering']):
    for j in range(beam_two.named_shape()['pointing']):
        ind = {
            "phase_steering":i,
            "pointing":j,
        }

        pyant.plotting.gain_heatmap(
            beam_two, 
            resolution=901, 
            min_elevation=20.0, 
            ax=axes[i,j],
            ind = ind,
        )

        #the boresight relative pointing is always in radians
        G0 = beam_two.gain_tf(theta=0, phi=np.radians(beam_two.phase_steering[i]), ind=ind)
        print(f'Pointing: az={beam_two.azimuth[j]} deg, el={beam_two.elevation[j]} deg, ' + \
              f'phase steering={beam_two.phase_steering[i]} deg ' + \
              f'-> Peak gain = {np.log10(G0)*10:8.3f} dB')

        #axes[i,j].axis('scaled')
        axes[i,j].set_title(f'{int(beam_two.phase_steering[i])} ph-st | {int(beam_two.elevation[j])} el')

pyant.plotting.show()
