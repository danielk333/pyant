#!/usr/bin/env python

'''Defines an antenna's or entire radar system's radiation pattern

(c) 2020 Daniel Kastinen
'''
import numpy as np
from abc import ABC, abstractmethod

from . import coordinates

class Beam(ABC):
    '''Defines the radiation pattern of a radar station.

    :param float frequency: Frequency of radiation pattern.
    :param float azimuth: Azimuth of pointing direction in dgreees.
    :param float elevation: Elevation of pointing direction in degrees.
    :param bool radians: If :code:`True` all input/output angles are in radians, else they are in degrees
    
    :ivar float frequency: Frequency of radiation pattern.
    :ivar float azimuth: Azimuth of pointing direction in dgreees.
    :ivar float elevation: Elevation of pointing direction in degrees.
    :ivar bool radians: If :code:`True` all input/output angles are in radians, else they are in degrees
    :ivar numpy.array pointing: Cartesian vector in local coordinates describing pointing direction.
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


    def copy(self):
        '''Return a copy of the current instance.
        '''
        return NotImplementedError('')


    def point(self, azimuth, elevation):
        '''Point beam towards azimuth and elevation coordinate.
        
            :param float azimuth: Azimuth east of north of pointing direction.
            :param float elevation: Elevation from horizon of pointing direction.
            :param bool radians: If :code:`True` all input/output angles are in radians, else they are in degrees
            :return: :code:`None`
        '''
        self.azimuth = azimuth
        self.elevation = elevation
        self.pointing = coordinates.sph_to_cart(
            np.array([azimuth, elevation, 1], dtype=np.float64),
            radians = self.radians,
        )


    def sph_point(self, k):
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
        

    def sph_angle(self, azimuth, elevation, radians=False):
        '''Get angle between azimuth and elevation and pointing direction.
        
            :param float azimuth: Azimuth east of north to measure from.
            :param float elevation: Elevation from horizon to measure from.
            :param bool radians: If :code:`True` all input/output angles are in radians, else they are in degrees
            
            :return: Angle between pointing and given direction.
            :rtype: float
        '''
        direction = coordinates.azel_to_cart(azimuth, elevation, 1.0, radians=radians)
        return coordinates.vector_angle(self.pointing, direction, radians=radians)

    def angle(self, k, radians=False):
        '''Get angle between local direction and pointing direction.
        
            :param numpy.array k: Direction to evaluate angle to.
            :param bool radians: If :code:`True` all input/output angles are in radians, else they are in degrees

            :return: Angle between pointing and given direction.
            :rtype: float
        '''
        return coordinates.vector_angle(self.pointing, k, radians=radians)

    @abstractmethod
    def gain(self, k):
        '''Return the gain in the given direction.

        :param numpy.array k: Direction in local coordinates to evaluate gain in.
        :return: Radar gain in the given direction.
        :rtype: float
        '''
        pass

    
    def sph_gain(self, azimuth, elevation, radians=False):
        '''Return the gain in the given direction.

        :param float azimuth: Azimuth east of north to evaluate gain in.
        :param float elevation: Elevation from horizon to evaluate gain in.
        :param bool radians: If :code:`True` all input/output angles are in radians, else they are in degrees
        :return: Radar gain in the given direction.
        :rtype: float
        '''
        k = coordinates.azel_to_cart(azimuth, elevation, 1.0, radians=radians)
        return self.gain(k)