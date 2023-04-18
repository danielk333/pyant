#!/usr/bin/env python
import numpy as np


def monte_carlo_sample_gain(beam, num, seed=None, vectorized=True):
    '''Perform uniform sampling on the positive hemisphere.
    '''

    if seed is not None:
        np.random.seed(seed)

    k = np.zeros((3, num), dtype=np.float64)
    k0 = np.random.rand(2, num)

    kxy2 = k0[0, :]**2 + k0[1, :]**2
    ind = np.sum(kxy2 <= 1)

    k[:2, :ind] = k0[:, kxy2 <= 1]

    while ind < num:
        kadd = np.random.rand(2)
        if kadd[0]**2 + kadd[1]**2 <= 1:
            k[:2, ind] = kadd
            ind += 1

    k[2, :] = np.sqrt(1.0 - k[0, :]**2 + k[1, :]**2)

    if vectorized:
        G = beam.gain(k)
    else:
        G = np.zeros((k.shape[1],))
        for i in range(len(G)):
            G[i] = beam.gain(k[:, i])

    return G, k


def monte_carlo_integrate_gain_hemisphere(beam, num, seed=None, vectorized=True):
    '''Calculate the integral of the gain pattern over the hemisphere using monte carlo integration.
    '''

    # sample surface is a unit circle in the x-y plane with area
    A = np.pi
    G, _ = monte_carlo_sample_gain(beam, num, seed=seed, vectorized=vectorized)

    # then the integral approximation is the mean of the sample function values times the area
    total_I = np.mean(G)*A

    return total_I
