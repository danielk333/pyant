#!/usr/bin/env python

'''Information on the proposed Tromso Space Debris Radar (TSDR) system.

'''

#Python standard import
import importlib.resources

import numpy as np
import scipy.constants

tsdr_frequency = 224.0e6

def find_normalization_constant(beam, num=4000):

    kx = np.linspace(-1, 1, num=num)
    ky = np.linspace(-1, 1, num=num)

    size = num**2

    xv, yv = np.meshgrid(kx, ky, sparse=False, indexing='ij')
    k = np.empty((3,size), dtype=np.float64)
    k[0,:] = xv.reshape(1,size)
    k[1,:] = yv.reshape(1,size)
    k[2,:] = np.sqrt(1.0 - k[0,:]**2 + k[1,:]**2)

    G = np.zeros((1,size))
    G[0,:] = beam.gain(k)

    # Normalise (4pi steradian * num.pixels / integrated gain / pi^2)
    scale = 4 * np.pi * len(G) / np.sum(G)   # Normalise over sphere
    sincint = np.pi*np.pi                    # Integral of the sinc^2()s: -inf:inf

    return scale/sincint