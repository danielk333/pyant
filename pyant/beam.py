#!/usr/bin/env python

'''Defines an antenna's or entire radar system's radiation pattern

(c) 2020 Daniel Kastinen
'''
import numpy as np
from abc import ABC, abstractmethod
import collections


import scipy.constants

from . import coordinates

class Beam(ABC):
    '''Defines the radiation pattern, i.e gain, of a radar. Gain here means amplification of the electromagnetic wave amplitude when transfered to a complex voltage amplitude.

    :param float frequency: Frequency of radiation pattern.
    :param float azimuth: Azimuth of pointing direction.
    :param float elevation: Elevation of pointing direction.
    :param bool radians: If :code:`True` all input/output angles are in radians, else they are in degrees
    
    :ivar float frequency: Frequency of radiation pattern.
    :ivar float azimuth: Azimuth of pointing direction.
    :ivar float elevation: Elevation of pointing direction.
    :ivar bool radians: If :code:`True` all input/output angles are in radians, else they are in degrees
    :ivar numpy.ndarray pointing: Cartesian vector in local coordinates describing pointing direction.
    '''


    def __init__(self, azimuth, elevation, frequency, radians=False, **kwargs):
        '''Basic constructor.
        '''
        if isinstance(frequency, list) or isinstance(frequency, tuple):
            frequency = np.array(frequency, dtype=np.float64)

        sph = Beam._azel_to_numpy(azimuth, elevation)

        self.frequency = frequency
        self.azimuth = sph[0,...]
        self.elevation = sph[1,...]
        self.radians = radians


        self.pointing = coordinates.sph_to_cart(sph, radians = radians)

    @staticmethod
    def _get_len(obj):
        if isinstance(obj, np.ndarray):
            if len(obj.shape) == 0:
                return None
            else:
                return obj.size
        elif isinstance(obj, list) or isinstance(obj, tuple):
            return len(obj)
        else:
            return None


    def named_shape(self):
        '''Return the named shape of all variables. This can be overridden to extend the possible variables contained in an instance.
        '''
        if len(self.pointing.shape) > 1:
            p_len = self.pointing.shape[1]
        else:
            p_len = None
        f_len = Beam._get_len(self.frequency)
        return collections.OrderedDict(pointing=p_len, frequency=f_len)


    @property
    def shape(self):
        '''The additional dimensions added to the output Gain.
        '''
        return tuple([x for x in self.named_shape().values() if x is not None])


    @property
    def variables(self):
        return tuple([k for k,x in self.named_shape().items() if x is not None])


    def get_parameters(self, ind):
        ind, shape = self.default_ind(ind)
        
        if shape['frequency'] is not None:
            frequency = self.frequency[ind['frequency']]
        else:
            frequency = self.frequency

        if shape['pointing'] is not None:
            pointing = self.pointing[:,ind['pointing']]
        else:
            pointing = self.pointing

        return frequency, pointing


    def default_ind(self, ind):
        if ind is None:
            ind = {}
        shape = self.named_shape()
        for key in shape:
            if key not in ind and shape[key] is not None:
                ind[key] = 0
        return ind, shape


    def _check_radians(self, azimuth, elevation, radians):
        if radians is None:
            return azimuth, elevation
        else:
            if radians == self.radians:
                return azimuth, elevation
            else:
                if radians:
                    return np.degrees(azimuth), np.degrees(elevation)
                else:
                    return np.radians(azimuth), np.radians(elevation)

    @staticmethod
    def _azel_to_numpy(azimuth, elevation):
        az_len = Beam._get_len(azimuth)
        el_len = Beam._get_len(elevation)

        if az_len is not None:
            shape = (3,az_len)
        elif el_len is not None:
            shape = (3,el_len)
        else:
            shape = (3,)

        sph = np.empty(shape, dtype=np.float64)
        sph[0,...] = azimuth
        sph[1,...] = elevation
        sph[2,...] = 1.0

        return sph


    @property
    def wavelength(self):
        return scipy.constants.c/self.frequency


    @wavelength.setter
    def wavelength(self, val):
        self.frequency = scipy.constants.c/val 


    def copy(self):
        '''Return a copy of the current instance.
        '''
        return NotImplementedError('')


    def complex(self, k, polarization, ind):
        '''The complex voltage output can be implemented as a middle step in gain calculation. Can include polarization channels.
        '''
        return NotImplementedError('')


    def sph_point(self, azimuth, elevation, radians=None):
        '''Point beam towards azimuth and elevation coordinate.
        
        :param float azimuth: Azimuth east of north of pointing direction.
        :param float elevation: Elevation from horizon of pointing direction.
        :param bool radians: If :code:`True` all input/output angles are in radians, if False degrees are used. Defaults to instance settings :code:`self.radians`.
        :return: :code:`None`

        '''
        azimuth, elevation = self._check_radians(azimuth, elevation, radians)
        sph = Beam._azel_to_numpy(azimuth, elevation)

        self.azimuth = azimuth
        self.elevation = elevation
        self.pointing = coordinates.sph_to_cart(sph, radians = self.radians)


    def point(self, k):
        '''Point beam in local cartesian direction.
        
        :param numpy.ndarray k: Pointing direction in local coordinates.
        :return: :code:`None`
        '''
        self.pointing = k/np.linalg.norm(k,axis=0)
        sph = coordinates.cart_to_sph(
            self.pointing,
            radians = self.radians,
        )
        self.azimuth = sph[0,...]
        self.elevation = sph[1,...]
        

    def sph_angle(self, azimuth, elevation, radians=None):
        '''Get angle between azimuth and elevation and pointing direction.
    
        :param float azimuth: Azimuth east of north to measure from.
        :param float elevation: Elevation from horizon to measure from.
        :param bool radians: If :code:`True` all input/output angles are in radians, if False degrees are used. Defaults to instance settings :code:`self.radians`.

        :return: Angle between pointing and given direction.
        :rtype: float
        '''
        if radians is None:
            radians = self.radians

        sph = Beam._azel_to_numpy(azimuth, elevation)
        k = coordinates.sph_to_cart(sph, radians=radians)
        return self.angle(k, radians=radians)

    def angle(self, k, radians=None):
        '''Get angle between local direction and pointing direction.
        
        :param numpy.ndarray k: Direction to evaluate angle to.
        :param bool radians: If :code:`True` all input/output angles are in radians, if False degrees are used. Defaults to instance settings :code:`self.radians`.

        :return: Angle between pointing and given direction.
        :rtype: float
        '''
        if radians is None:
            radians = self.radians

        if len(self.pointing.shape) > 1:
            if len(k.shape) > 1:
                theta = np.empty((k.shape[1], self.pointing.shape[1]), dtype=k.dtype)
            else:
                theta = np.empty((self.pointing.shape[1], ), dtype=k.dtype)

            for ind in range(self.pointing.shape[1]):
                theta[...,ind] = coordinates.vector_angle(self.pointing[ind,:], k, radians=radians)
        else:
            theta = coordinates.vector_angle(self.pointing, k, radians=radians)

        return theta


    @abstractmethod
    def gain(self, k, polarization=None, ind=None):
        '''Return the gain in the given direction. This method should be vectorized.

        :param numpy.ndarray k: Direction in local coordinates to evaluate gain in. Must be a `(3,)` vector or a `(3,n)` matrix.
        :param numpy.ndarray polarization: The Jones vector of the incoming plane waves, if applicable for the beam in question.
        :param dict ind: The incidences of the parameters that are variables. Defaults to the first element of each parameter. If e.g. pointing is a variable with 5 values, :code:`ind={'pointing':2}` would evaluate the gain using the third pointing direction.

        :return: Radar gain in the given direction. If input is a `(3,)` vector, output is a float. If input is a `(3,n)` matrix output is a `(n,)` vector of gains.
        :rtype: float/numpy.ndarray
        '''
        pass

    
    def sph_gain(self, azimuth, elevation, polarization=None, radians=None, ind=None):
        '''Return the gain in the given direction.

        :param float azimuth: Azimuth east of north to evaluate gain in.
        :param float elevation: Elevation from horizon to evaluate gain in.
        :param bool radians: If :code:`True` all input/output angles are in radians, if False degrees are used. Defaults to instance settings :code:`self.radians`.

        :return: Radar gain in the given direction.
        :rtype: float
        '''
        if radians is None:
            radians = self.radians

        sph = Beam._azel_to_numpy(azimuth, elevation)

        k = coordinates.sph_to_cart(sph, radians=radians)
        return self.gain(k, polarization=polarization, ind = ind)