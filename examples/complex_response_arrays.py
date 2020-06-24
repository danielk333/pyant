'''
Complex response of array
===========================
'''
import time
import numpy as np

import pyant

ant_n = 55
dr = 2.0

xv, yv = np.meshgrid(
    np.arange(-ant_n//2,ant_n//2)*dr, 
    np.arange(-ant_n//2,ant_n//2)*dr, 
)
antennas = np.zeros((3, ant_n**2))
antennas[0,:] = xv.flatten()
antennas[1,:] = yv.flatten()

ant = pyant.Array(
    azimuth=0,
    elevation=90.0, 
    frequency=46.5e6,
    antennas=antennas,
)

ant_lin = pyant.Array(
    azimuth=0,
    elevation=90.0, 
    frequency=46.5e6,
    polarization=np.array([1,0]),
    antennas=antennas,
)

k = np.array([0,0,1])
km = np.array([[0,0,1],[0,0.1,0.9],[0,0.1,0.8]]).T


print(f'Gain LHCP: {ant.gain(k)}')
print(f'Gain LHCP: {ant.gain(km)}')

print(f'Gain |H>: {ant_lin.gain(k)}')
print(f'Gain |H>: {ant_lin.gain(km)}')

print(f'Complex response from {k} of LHCP analysis when signal is |H>: {ant.complex(k=k, polarization=np.array([1,0]))}')
print(f'Complex response from {k} of LHCP analysis when signal is LHCP: {ant.complex(k=k, polarization=ant.polarization)}')
