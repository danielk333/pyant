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
    '''Gain pattern of an antenna array radar receiving/transmitting plane waves. Assumes the same antenna is used throughout the array.

    Antennas can be combined into a single channel of multiple depending on the shape of the input :code:`antenna` ndarray.

    :param numpy.ndarray antennas: `(3, n)` or `(3, n, c)` numpy array of antenna spatial positions, where `n` is the number of antennas and `c` is the number of sub-arrays.
    :param float scaling: Scaling parameter for the output gain, can be interpreted as an antenna element scalar gain.

    :ivar numpy.ndarray antennas: `(3, n)` or `(3, n, c)` numpy array of antenna spatial positions, where `n` is the number of antennas and `c` is the number of sub-arrays.
    :ivar float scaling: Scaling parameter for the output gain, can be interpreted as an antenna element scalar gain.
    :ivar int channels: Number of sub-arrays the antenna array has, i.e the number of channels.

    '''
    def __init__(self, azimuth, elevation, frequency, antennas, scaling=1.0, **kwargs):
        super().__init__(azimuth, elevation, frequency, **kwargs)
        assert len(antennas.shape) in [2,3]
        assert antennas.shape[0] == 3
        assert isinstance(antennas, np.ndarray)

        if len(antennas.shape) == 2:
            antennas = antennas.reshape(antennas.shape[0], antennas.shape[1], 1)
        antennas = np.transpose(antennas, (1, 0, 2))

        self.antennas = antennas
        self.scaling = scaling


    @property
    def channels(self):
        '''Number of channels returned by complex output.
        '''
        return self.antennas.shape[2]


    def antenna_element(self, k):
        '''Antenna element gain pattern, projected aperture approximation based on pointing.
        '''
        return k[2]*self.scaling


    def gain(self, k):
        '''Gain of the antenna array.
        '''
        G = self.complex(k, channels=None)
        return np.abs(np.sum(G,axis=0))


    def complex(self, k, channels=None):
        '''Complex voltage output signal after summation of antennas.
        '''
        inds = np.arange(self.channels, dtype=np.int)
        if channels is not None:
            inds[channels] = True

        chan_num = len(inds)

        k_ = k/np.linalg.norm(k, axis=0)
        if len(k.shape) == 1:
            G = np.zeros((chan_num, ), dtype=np.complex128)
            p = self.pointing
        else:
            G = np.zeros((chan_num, k.shape[1]), dtype=np.complex128)
            p = np.repeat(self.pointing.reshape(3,1),k.shape[1], axis=1)
        wavelength = self.wavelength

        #r in meters, divide by lambda
        for i in range(chan_num):
            for r in self.antennas[:,:,inds[i]]:
                G[i,...] += plane_wave(k_,r/wavelength,p)

        return G*self.antenna_element(k_)

