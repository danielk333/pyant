#!/usr/bin/env python

'''Defines an antenna's or entire radar system's radiation pattern

(c) 2020 Daniel Kastinen
'''
import numpy as np
from abc import ABC, abstractmethod

import scipy.constants

from . import coordinates

class Beam(ABC):
    '''Defines the radiation pattern of a radar station.

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
        self.frequency = frequency
        self.azimuth = azimuth
        self.elevation = elevation
        self.radians = radians
        self.pointing = coordinates.sph_to_cart(
            np.array([azimuth, elevation, 1]),
            radians = radians,
        )


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
        if isinstance(azimuth, np.ndarray):
            sph = np.ones((3,len(azimuth)), dtype=azimuth.dtype)
        elif isinstance(elevation, np.ndarray):
            sph = np.ones((3,len(elevation)), dtype=elevation.dtype)
        else:
            sph = np.empty((3,), dtype=np.float64)

        sph[0,...] = azimuth
        sph[1,...] = elevation
        sph[2,...] = 1.0

        return sph


    @property
    def wavelength(self):
        return scipy.constants.c/self.frequency


    def copy(self):
        '''Return a copy of the current instance.
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

        self.azimuth = azimuth
        self.elevation = elevation
        self.pointing = coordinates.sph_to_cart(
            np.array([self.azimuth, self.elevation, 1], dtype=np.float64),
            radians = self.radians,
        )


    def point(self, k):
        '''Point beam in local cartesian direction.
        
        :param numpy.ndarray k: Pointing direction in local coordinates.
        :return: :code:`None`
        '''
        self.pointing = k/np.linalg.norm(k)
        sph = coordinates.cart_to_sph(
            self.pointing,
            radians = self.radians,
        )
        self.azimuth = sph[0]
        self.elevation = sph[1]
        

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

        direction = coordinates.sph_to_cart(sph, radians=radians)
        return coordinates.vector_angle(self.pointing, direction, radians=radians)

    def angle(self, k, radians=None):
        '''Get angle between local direction and pointing direction.
        
        :param numpy.ndarray k: Direction to evaluate angle to.
        :param bool radians: If :code:`True` all input/output angles are in radians, if False degrees are used. Defaults to instance settings :code:`self.radians`.

        :return: Angle between pointing and given direction.
        :rtype: float
        '''
        if radians is None:
            radians = self.radians

        return coordinates.vector_angle(self.pointing, k, radians=radians)

    @abstractmethod
    def gain(self, k):
        '''Return the gain in the given direction. This method should be vectorized.

        :param numpy.ndarray k: Direction in local coordinates to evaluate gain in. Must be a `(3,)` vector or a `(3,n)` matrix.
        :return: Radar gain in the given direction. If input is a `(3,)` vector, output is a float. If input is a `(3,n)` matrix output is a `(n,)` vector of gains.
        :rtype: float/numpy.ndarray
        '''
        pass

    
    def sph_gain(self, azimuth, elevation, radians=None):
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
        return self.gain(k)