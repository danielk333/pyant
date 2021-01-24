#!/usr/bin/env python

import copy
import time

import numpy as np
import scipy.constants
import scipy.special


REQUIRED_FUNCTIONALITY = {
    'k_vectorized_gain': True,
    'supports_registered_parameters': True,
    'efficient_vectorized_gain': False,
    'complex_gain': False,
    'copyable': False,
    'vectorized_parameters': False,
    'normalized_gain': False,
}

def validate_functionality(beam):
    '''Checks the functionality of the 
    '''
    DETECTED_FUNCTIONALITY = {key:False for key in REQUIRED_FUNCTIONALITY}
    PROFILE = {}

    kn = 100

    dk = 2.0/float(kn)

    kx = np.linspace(-1, 1, num=kn)
    ky = np.linspace(-1, 1, num=kn)

    size = len(kx)*len(ky)

    #loop version
    start_time = time.time()

    G = np.zeros((len(kx),len(ky)))
    cnt_ = 0
    for i,x in enumerate(kx):
        for j,y in enumerate(ky):
            xy2 = x**2 + y**2
            if xy2 <= 1:
                cnt_ += 1
                k=np.array([x, y, np.sqrt(1.0 - xy2)])
                G[i,j] = beam.gain(k)

    loop_time = time.time() - start_time
    PROFILE['loop_gain_time'] = loop_time/cnt_

    #vectorized version
    start_time = time.time()

    xv, yv = np.meshgrid(kx, ky, sparse=False, indexing='ij')
    k = np.empty((3,size), dtype=np.float64)
    k[0,:] = xv.reshape(1,size)
    k[1,:] = yv.reshape(1,size)
    k = k[:,k[0,:]**2 + k[1,:]**2 <= 1]
    k[2,:] = np.sqrt(1.0 - k[0,:]**2 + k[1,:]**2)

    #We want to use reshape as a inverse function so we make sure its the exact same dimensionality
    G = np.zeros((1,k.shape[1]))
    k_vectorized_gain = True
    try:
        G[0,:] = beam.gain(k)
    except:
        k_vectorized_gain = False
    
    DETECTED_FUNCTIONALITY['k_vectorized_gain'] = k_vectorized_gain

    vector_time = time.time() - start_time
    PROFILE['vectorized_gain_time'] = vector_time/cnt_

    if vector_time < loop_time:
        if k_vectorized_gain:
            DETECTED_FUNCTIONALITY['efficient_vectorized_gain'] = True

    #todo: profile complex time

    complex_gain = True
    try:
        G_ = beam.complex(k[:,0], polarization=np.array([1,0]), ind=None)
    except:
        complex_gain = False

    DETECTED_FUNCTIONALITY['complex_gain'] = complex_gain

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

    supports_registered_parameters = False
    vectorized_parameters = False

    #todo: profile parameters time complexity

    #TODO: rework parameters again

    #TODO: do MC integration instead

    int_aprox = np.sum(G*dk*dk)

    if int_aprox < 1.1:
        DETECTED_FUNCTIONALITY['normalized_gain'] = True
    else:
        DETECTED_FUNCTIONALITY['normalized_gain'] = False

    return DETECTED_FUNCTIONALITY, PROFILE



