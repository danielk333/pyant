#!/usr/bin/env python

import copy
import time

import numpy as np
import scipy.constants
import scipy.special

from .functions import monte_carlo_integrate_gain_hemisphere

REQUIRED_FUNCTIONALITY = {
    'k_vectorized_gain': True,
    'supports_registered_parameters': True,
    'efficient_vectorized_gain': False,
    'complex_signal': False,
    'k_vectorized_complex_signal': False,
    'copyable': False,
    'normalized_gain': False,
}


def _generate_kG_grids(grid_size):
    '''Generate a grid over all xy k vector projections and a empty vector for gains.
    '''
    kx = np.linspace(-1, 1, num=grid_size)
    ky = np.linspace(-1, 1, num=grid_size)
    size = len(kx)*len(ky)

    xv, yv = np.meshgrid(kx, ky, sparse=False, indexing='ij')
    k = np.empty((3,size), dtype=np.float64)
    k[0,:] = xv.reshape(1,size)
    k[1,:] = yv.reshape(1,size)

    kxy2 = k[0,:]**2 + k[1,:]**2

    k = k[:,kxy2 <= 1]

    k[2,:] = np.sqrt(1.0 - k[0,:]**2 + k[1,:]**2)

    G = np.zeros((k.shape[1],))

    return k, G


def _calculate_k_grid_loop(func, grid_size):
    '''Calculate function over a full grid over xy k vectors using a loop
    '''

    k, G = _generate_kG_grids(grid_size)
    G = G.tolist()

    start_time = time.time()

    for i in range(len(G)):
        G[i] = func(k[:,i])

    loop_time = time.time() - start_time

    return G, k, loop_time


def _calculate_k_grid_vectorized(func, grid_size):
    '''Calculate function over a full grid over xy k vectors using vectorization
    '''

    k, _ = _generate_kG_grids(grid_size)

    start_time = time.time()

    G = func(k)

    vector_time = time.time() - start_time

    return G, k, vector_time



def validate_functionality(beam, MC_num = 1000, grid_size=100):
    '''Checks the functionality of the 
    '''
    DETECTED_FUNCTIONALITY = {key:False for key in REQUIRED_FUNCTIONALITY}
    PROFILE = {}

    dk = 2.0/float(grid_size)

    G, k, loop_time = _calculate_k_grid_loop(beam.gain, grid_size)

    grid_points = len(G)
    PROFILE['loop_gain_time'] = loop_time/grid_points


    k_vectorized_gain = True
    try:
        _, _, vector_time = _calculate_k_grid_vectorized(beam.gain, grid_size)
    except:
        vector_time = np.inf
        k_vectorized_gain = False
    
    DETECTED_FUNCTIONALITY['k_vectorized_gain'] = k_vectorized_gain
    PROFILE['vectorized_gain_time'] = vector_time/grid_points


    if vector_time < loop_time:
        if k_vectorized_gain:
            DETECTED_FUNCTIONALITY['efficient_vectorized_gain'] = True


    complex_signal = True
    try:
        _, _, c_loop_time = _calculate_k_grid_loop(beam.complex, grid_size)
    except:
        c_loop_time = np.inf
        complex_signal = False
    DETECTED_FUNCTIONALITY['complex_signal'] = complex_signal
    PROFILE['loop_complex_time'] = c_loop_time/grid_points


    k_vectorized_complex_signal = True
    try:
        _, _, c_vec_time = _calculate_k_grid_vectorized(beam.complex, grid_size)
    except:
        c_vec_time = np.inf
        k_vectorized_complex_signal = False
    DETECTED_FUNCTIONALITY['k_vectorized_complex_signal'] = k_vectorized_complex_signal
    PROFILE['vectorized_complex_time'] = c_vec_time/grid_points


    copyable = True
    try:
        beam2 = beam.copy()
    except:
        beam2 = beam
        copyable = False

    if beam2 is beam:
        copyable = False
    if beam2.pointing is beam.pointing:
        copyable = False

    DETECTED_FUNCTIONALITY['copyable'] = copyable

    # vectorized_parameters = False

    supports_registered_parameters = True

    try:
        for key, val in zip(beam.parameters, beam.get_parameters(ind=None)):
            G_ = beam.gain(k[:,0], ind={key:val})
    except:
        supports_registered_parameters = False

    DETECTED_FUNCTIONALITY['supports_registered_parameters'] = supports_registered_parameters

    I = monte_carlo_integrate_gain_hemisphere(
        beam, 
        MC_num, 
        vectorized=DETECTED_FUNCTIONALITY['k_vectorized_gain'],
    )

    PROFILE['gain_integral'] = I

    #about 10% accuracy
    if np.abs(I - 1) < 0.1:
        DETECTED_FUNCTIONALITY['normalized_gain'] = True
    else:
        DETECTED_FUNCTIONALITY['normalized_gain'] = False

    return DETECTED_FUNCTIONALITY, PROFILE



