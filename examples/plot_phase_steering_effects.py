'''
Phase steerable Half-pipe
============================
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import pyant

num = 1000
num_ph = 300
levels = 25

#phase steering
phi0 = np.linspace(-90, 90, num_ph)

beam = pyant.PhasedFiniteCylindricalParabola(
    azimuth=0,
    elevation=90.0, 
    depth=18,
    phase_steering = 0.0,
    frequency=30e6,
    width=120.0,
    height=40.0,
)

G = np.zeros((num_ph,num), dtype=np.float64)

k = np.zeros((3,num), dtype=np.float64)
#all "along feed bridge K vectors"
k[0,:] = np.linspace(-1, 1, num)
k[2,:] = np.sqrt(1 - k[0,:]**2 - k[1,:]**2)


for i in range(num_ph):
    #set phase
    beam.phase_steering = phi0[i]
    G[i,:] = beam.gain(k).flatten()

SdB = np.log10(G)*10.0

fig, ax = plt.subplots(1,1,figsize=(10,6),dpi=80)

bins = MaxNLocator(nbins=levels).tick_values(-50, np.nanmax(SdB))
cmap = plt.get_cmap('plasma')
norm = BoundaryNorm(bins, ncolors=cmap.N, clip=True)

conf = ax.pcolormesh(np.degrees(np.arcsin(k[0,:])), phi0, SdB, cmap=cmap, norm=norm) #, vmin=0, vmax=np.nanmax(SdB)

plt.colorbar(conf)

plt.show()
