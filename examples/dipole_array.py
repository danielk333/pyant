'''
Dipole array
=============
'''
import numpy as np
import matplotlib.pyplot as plt

import pyant

xv, yv = np.meshgrid(np.linspace(-50, 50, num=22),
                     np.linspace(-50, 50, num=22))
antennas = np.zeros((3, 22**2))
antennas[0, :] = xv.flatten()
antennas[1, :] = yv.flatten()

beam = pyant.DipoleArray(
    azimuth=0,
    elevation=90.0, 
    frequency=46.5e6,
    antennas=antennas,
    antenna_rotation=45.0,
)

pols = [
    (np.array([1, 1j])/np.sqrt(2), 'RHCP'),
    (np.array([1, 0]), 'Vertical'),
]

fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=80)
for pol, ax in zip(pols, axes.flatten()):
    jones, name = pol
    pyant.plotting.gain_heatmap(
        beam, polarization = jones, ax=ax, min_elevation=80)
    ax.set_title(f'Incoming Jones={name}', fontsize=22)

plt.suptitle(f'Square single-dipole antenna array @ {beam.antenna_rotation} degrees', fontsize=18)

pyant.plotting.show()
