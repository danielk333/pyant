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

#num = 1000
#num_ph = 300


def steering_gain(beam, num_ph=701, num_az=901, symmetric=True, corrected=True):
    #phase steering

    G = np.zeros((num_ph,num_az), dtype=np.float64)

    k = np.zeros((3,num_az), dtype=np.float64)
    #all "along feed bridge K vectors"
    phi0 = np.linspace(-90*symmetric, 90, num_ph)
    k[0,:] = np.linspace(-1, 1, num_az)
    k[2,:] = np.sqrt(1 - k[0,:]**2 - k[1,:]**2)

    for i in range(num_ph):
        #set phase
        if corrected:
            beam.phase_steering = beam._nominal_phase_steering(phi0[i])
        else:
            beam.phase_steering = phi0[i]
        G[i,:] = beam.gain(k).flatten()

    return k, phi0, G


def make_steering_plot(k, phi0, G, levels=14, symmetric=None, corrected=True, ax=None):
    # Symmetric=None for no 'true k' dashed line
    # Symmetric=True plots from min(kx) to max(kx)
    # Symmetric=False plots from 0 to max(kx)
    SdB = np.log10(G)*10.0

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(10,6),dpi=80)

    bins = MaxNLocator(nbins=levels).tick_values(-8, np.nanmax(SdB))
    cmap = plt.get_cmap('plasma')
    norm = BoundaryNorm(bins, ncolors=cmap.N, clip=True)
    kx = np.degrees(np.arcsin(k[0]))

    conf = ax.pcolormesh(kx, phi0, SdB, cmap=cmap, norm=norm, shading='auto') #, vmin=0, vmax=np.nanmax(SdB)
    plt.colorbar(conf, ax=ax, fraction=0.04, pad=0.01)

    if symmetric is not None:
        k_true = [min(kx)*symmetric, max(kx)]
        ax.plot(k_true, [min(phi0), max(phi0)], 'w--')


def steering_plot(beam, num_ph=301, num_az=501, levels=14, symmetric=True, corrected=True, ax=None):

    k, phi0, G = steering_gain(beam, num_ph=num_ph, num_az=num_az, symmetric=symmetric, corrected=corrected)

    make_steering_plot(k, phi0, G, levels=levels, symmetric=symmetric, corrected=corrected, ax=ax)





if __name__ == '__main__':

    beam = pyant.PhasedFiniteCylindricalParabola(
        azimuth=0,
        elevation=90.0,
        depth=18,
        phase_steering = 0.0,
        frequency=300e6,
        width=120.0,
        height=40.0,
    )

    fig, ax = plt.subplots(1, 2, figsize=(10,6), sharex='all', sharey='all')

    steering_plot(beam, symmetric=False, corrected=False, ax=ax[0])
    steering_plot(beam, symmetric=False, corrected=True, ax=ax[1])

    ax[0].set_title('Phase steering (nominal)')
    ax[1].set_title('Phase steering (corrected)')

    plt.show()
