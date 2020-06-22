#!/usr/bin/env python

'''Useful coordinate related functions.

(c) 2020 Daniel Kastinen
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def show():
    '''Shorthand for matplotlib :code:`show` function.'''
    plt.show()


def gain_heatmap(beam, resolution=201, min_elevation=0.0, levels=20, ax=None, vectorized=True):
    '''Creates a heatmap of the beam-patters as a function of azimuth and elevation in terms of wave vector ground projection coordinates.
    
    :param Beam beam: Beam pattern to plot.
    :param int resolution: Number of points to divide the wave vector x and y components into, total number of calculation points is the square of this number.
    :param float min_elevation: Minimum elevation in degrees, elevation range is from this number to :math:`90^\circ`. This number defines the half the length of the square that the gain is calculated over, i.e. :math:`\cos(el_{min})`.
    :param int levels: Number of levels in the contour plot.
    :param bool vectorized: Use vectorized gain functionality to calculate gain-map.
    :return: matplotlib axis and figure handles
    '''

    #turn on TeX interperter
    plt.rc('text', usetex=True)

    if ax is None:
        fig = plt.figure(figsize=(15,7))
        ax = fig.add_subplot(111)
    else:
        fig = None


    kx=np.linspace(
        beam.pointing[0] - np.cos(min_elevation*np.pi/180.0),
        beam.pointing[0] + np.cos(min_elevation*np.pi/180.0),
        num=resolution,
    )
    ky=np.linspace(
        beam.pointing[1] - np.cos(min_elevation*np.pi/180.0),
        beam.pointing[1] + np.cos(min_elevation*np.pi/180.0),
        num=resolution,
    )
    
    
    K=np.zeros((resolution,resolution,2))

    if vectorized:
        K[:,:,0], K[:,:,1] = np.meshgrid(kx, ky, sparse=False, indexing='ij')
        size = resolution**2
        k = np.empty((3,size), dtype=np.float64)
        k[0,:] = K[:,:,0].reshape(1,size)
        k[1,:] = K[:,:,1].reshape(1,size)

        z2 = k[0,:]**2 + k[1,:]**2
        z2_c = (beam.pointing[0] - k[0,:])**2 + (beam.pointing[1] - k[1,:])**2
        inds_ = np.logical_and(z2_c < np.cos(min_elevation*np.pi/180.0)**2, z2 <= 1.0)
        not_inds_ = np.logical_not(inds_)

        k[2,inds_] = np.sqrt(1.0 - z2[inds_])
        k[2,not_inds_] = 0
        S = np.zeros((1,size))
        S[0,inds_] = beam.gain(k[:,inds_])
        S = S.reshape(resolution,resolution)

    else:
        S = np.zeros((resolution,resolution))
        for i,x in enumerate(kx):
            for j,y in enumerate(ky):
                z2_c = (beam.pointing[0]-x)**2 + (beam.pointing[1]-y)**2
                z2 = x**2 + y**2
                if z2_c < np.cos(min_elevation*np.pi/180.0)**2 and z2 <= 1.0:

                    k=np.array([x, y, np.sqrt(1.0 - z2)])
                    S[i,j]=beam.gain(k)
                K[i,j,0]=x
                K[i,j,1]=y
    
    SdB = np.log10(S)*10.0
    SdB[np.isinf(SdB)] = 0
    SdB[np.isnan(SdB)] = 0
    SdB[SdB < 0] = 0
    conf = ax.contourf(K[:,:,0], K[:,:,1], SdB, cmap=cm.plasma, vmin=0, vmax=np.max(SdB), levels=levels)
    ax.set_xlabel('$k_x$ [1]',fontsize=24)
    ax.set_ylabel('$k_y$ [1]',fontsize=24)
    
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    cbar = plt.colorbar(conf, ax=ax)
    cbar.ax.set_ylabel('Gain $G$ [dB]',fontsize=24)
    ax.set_title('Gain pattern', fontsize=24)

    return fig, ax
