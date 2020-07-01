#!/usr/bin/env python

import numpy as np
import scipy.constants
import scipy.special

from .beam import Beam
from . import coordinates


def plane_wave(k, r, p, J):
    '''The complex plane wave function.

    :param numpy.ndarray k: Wave-vectors (wave propagation directions)
    :param numpy.ndarray r: Spatial locations (Antenna positions in space)
    :param numpy.ndarray p: Beam-forming direction by phase offsets (antenna array "pointing" direction)
    :param numpy.ndarray J: Polarization given as a Jones vector
    '''

    #in this, rows are antennas and columns are wave directions
    spat_wave = np.exp(1j*np.pi*2.0*np.dot(r,k-p))

    #broadcast to polarizations
    wave = spat_wave[..., None]*J[None, None, :]

    return wave


class Array(Beam):
    '''Gain pattern of an antenna array radar receiving/transmitting plane waves, i.e in the far field approximation regime. Assumes the same antenna is used throughout the array.

    Antennas can be combined into a single channel or multiple depending on the shape of the input :code:`antenna` ndarray.

    :param numpy.ndarray antennas: `(3, n)` or `(3, n, c)` numpy array of antenna spatial positions, where `n` is the number of antennas and `c` is the number of sub-arrays.
    :param float scaling: Scaling parameter for the output gain, can be interpreted as an antenna element scalar gain.
    :param numpy.ndarray polarization: The Jones vector of the assumed polarization used when calculating the gain. Default is Left-hand circular polarized.

    :ivar numpy.ndarray antennas: `(3, n)` or `(3, n, c)` numpy array of antenna spatial positions, where `n` is the number of antennas and `c` is the number of sub-arrays.
    :ivar float scaling: Scaling parameter for the output gain, can be interpreted as an antenna element scalar gain.
    :param numpy.ndarray polarization: The Jones vector of the assumed polarization used when calculating the gain.
    :ivar int channels: Number of sub-arrays the antenna array has, i.e the number of channels.

    '''
    def __init__(self, 
            azimuth, 
            elevation, 
            frequency, 
            antennas, 
            polarization=np.array([1, 1j])/np.sqrt(2), 
            scaling=1.0, 
            **kwargs
        ):
        super().__init__(azimuth, elevation, frequency, **kwargs)
        assert len(antennas.shape) in [2,3]
        assert antennas.shape[0] == 3
        assert isinstance(antennas, np.ndarray)

        if len(antennas.shape) == 2:
            antennas = antennas.reshape(antennas.shape[0], antennas.shape[1], 1)
        antennas = np.transpose(antennas, (1, 0, 2))

        if not np.all(np.iscomplex(polarization)):
            polarization = polarization.astype(np.complex128)

        self.antennas = antennas
        self.scaling = scaling
        self.polarization = polarization


    def copy(self):
        '''Return a copy of the current instance.
        '''
        return Array(
            frequency = self.frequency,
            azimuth = self.azimuth,
            elevation = self.elevation,
            radians = self.radians,
            antennas = np.transpose(self.antennas, (1, 0, 2)).copy(),
            scaling = self.scaling,
            polarization = self.polarization.copy(),
        )

    @property
    def channels(self):
        '''Number of channels returned by complex output.
        '''
        return self.antennas.shape[2]


    def antenna_element(self, k, polarization):
        '''Antenna element gain pattern, azimuthally symmetric dipole response.
        '''
        ret = np.ones(polarization.shape, dtype=k.dtype)
        return ret[:,None]*k[2,:]*self.scaling


    def gain(self, k, polarization=None):
        '''Gain of the antenna array.
        '''
        if polarization is None:
            polarization = self.polarization
        elif not np.all(np.iscomplex(polarization)):
            polarization = polarization.astype(np.complex128)

        G = self.complex(k, polarization, channels=None)

        lin_pol_check = np.abs(self.polarization) < 1e-6
        if not np.any(lin_pol_check):

            pol_comp = self.polarization[0]*self.polarization[1].conj()
            pol_comp = pol_comp.conj()/np.abs(pol_comp)
            G[:,0,...] *= pol_comp #align polarizations

        G = np.sum(G,axis=1) #coherent intergeneration over polarization
        G = np.sum(G,axis=0) #coherent intergeneration over channels
        return np.abs(G)


    def complex(self, k, polarization, channels=None):
        '''Complex voltage output signal after summation of antennas.

        :return: `(c,2,num_k)` ndarray where `c` is the number of channels requested, `2` are the two polarization axis of the Jones vector and `num_k` is the number of input wave vectors. If `num_k = 1` the returned ndarray is `(c,2)`.
        '''
        inds = np.arange(self.channels, dtype=np.int)
        if channels is not None:
            inds = inds[channels]

        chan_num = len(inds)

        k_ = k/np.linalg.norm(k, axis=0)
        if len(k.shape) == 1:
            psi = np.zeros((chan_num, 2, 1, ), dtype=np.complex128)
            p = self.pointing.reshape(3,1)
            k_ = k_.reshape(3,1)
        else:
            psi = np.zeros((chan_num, 2, k.shape[1], ), dtype=np.complex128)
            p = np.repeat(self.pointing.reshape(3,1),k.shape[1], axis=1)

        wavelength = self.wavelength

        #r in meters, divide by lambda
        for i in range(chan_num):
            subg_response = plane_wave(k_, self.antennas[:,:,inds[i]]/wavelength, p, polarization)
            psi[i,:,...] = subg_response.sum(axis=0).T

        ant_response = self.antenna_element(k_, polarization)

        psi[:,0,...] *= ant_response[None,0,...]
        psi[:,1,...] *= ant_response[None,1,...]

        if len(k.shape) == 1:
            psi = psi.reshape(psi.shape[:-1])


        return psi


class DipoleArray(Array):
    '''Array of dipole antennas.

    :param float antenna_rotation: Azimuth east of north of the dipole axis.

    :ivar float antenna_rotation: Azimuth east of north of the dipole axis.
    '''

    def __init__(self, 
            azimuth, 
            elevation, 
            frequency, 
            antennas, 
            polarization=np.array([1, 0]), 
            scaling=1.0, 
            antenna_rotation = 0.0,
            **kwargs
        ):
        super().__init__(
            azimuth, 
            elevation, 
            frequency, 
            antennas, 
            polarization, 
            scaling, 
            **kwargs
        )
        self.antenna_rotation = antenna_rotation

    def antenna_element(self, k, polarization):
        '''Theoretical gain pattern of a single dipole antenna.
        '''
        if not self.radians:
            antr = np.radians(self.antenna_rotation)
        else:
            antr = self.antenna_rotation

        az = np.pi/2 - np.arctan2(k[1,...],k[0,...])

        ret = np.ones((2, k.shape[1]), dtype=np.complex128)
        ret[0,...] = k[2,:]*(np.sin(az - antr)*polarization[0] + np.cos(az - antr)*polarization[1])
        ret[1,...] = 0.0
        return ret*self.scaling
