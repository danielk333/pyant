
#!/usr/bin/env python

import numpy as np
import scipy.constants
import scipy.special

from .beam import Beam
from . import coordinates


def plane_wave(k,r,p):
    '''The complex plane wave function.

    :param numpy.ndarray k: Wave-vector (wave propagation direction)
    :param numpy.ndarray r: Spatial location (Antenna position in space)
    :param numpy.ndarray p: Beam-forming direction (antenna array "pointing" direction)
    '''
    return np.exp(1j*np.pi*2.0*np.dot(r,k-p))


class Array(Beam):
    '''
    :param numpy.ndarray antennas: `(n, 3)` matrix of antenna spatial positions, where `n` is the number of antennas.
    :param float scaling: Scaling parameter for the output gain, can be interpreted as an antenna element scalar gain.

    :ivar numpy.ndarray antennas: `(n, 3)` matrix of antenna spatial positions, where `n` is the number of antennas.
    :ivar float scaling: Scaling parameter for the output gain, can be interpreted as an antenna element scalar gain.

    '''
    def __init__(self, azimuth, elevation, frequency, antennas, scaling=1.0, **kwargs):
        super().__init__(azimuth, elevation, frequency, **kwargs)
        self.antennas = antennas
        self.scaling = scaling


    def antenna_element(self, k):
        '''Antenna element gain pattern
        '''
        return self.pointing[2]*self.scaling


    def gain(self, k):
        k_ = k/np.linalg.norm(k, axis=0)
        if len(k.shape) == 1:
            G = np.exp(1j)*0.0
            p = self.pointing
        else:
            G = np.zeros((k.shape[1],), dtype=np.complex128)
            p = np.repeat(self.pointing.reshape(3,1),k.shape[1], axis=1)
        wavelength = self.wavelength

        #r in meters, divide by lambda
        for r in self.antennas:
            G += plane_wave(k_,r/wavelength,p)

        return np.abs(G.conj()*G)*self.antenna_element(k_)
