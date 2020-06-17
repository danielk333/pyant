#!/usr/bin/env python

'''Defines an antenna's or entire radar system's radiation pattern

(c) 2020- Daniel Kastinen
'''
import copy

import numpy as np
from abc import ABC, abstractmethod

from . import coordinates

class Beam(ABC):
    '''Defines the radiation pattern of a radar station.

    :param float frequency: Frequency of radiation pattern.
    :param float azimuth: Azimuth of pointing direction in dgreees.
    :param float elevation: Elevation of pointing direction in degrees.
    :attr numpy.array pointing: Cartesian vector in local coordinates describing pointing direction.
    

    :ivar float frequency: Frequency of radiation pattern.
    :ivar float azimuth: Azimuth of pointing direction in dgreees.
    :ivar float elevation: Elevation of pointing direction in degrees.
    :ivar numpy.array pointing: Cartesian vector in local coordinates describing pointing direction.
    '''


    def __init__(self, azimuth, elevation, frequency, radians=False, **kwargs):

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
        
            :param float azimuth: Azimuth of pointing direction in dgreees east of north.
            :param float elevation: Elevation of pointing direction in degrees from horizon.
        '''
        self.azimuth = azimuth
        self.elevation = elevation
        self.pointing = coordinates.sph_to_cart(
            np.array([azimuth, elevation, 1], dtype=np.float64),
            radians = self.radians,
        )


    def vector_point(self, k):
        '''Point beam in local cartesian direction.
        
            :param numpy.ndarray k: Pointing direction in local coordinates.
        '''
        self.pointing = k/np.linalg.norm(k)
        sph = coordinates.cart_to_sph(
            self.pointing,
            radians = self.radians,
        )
        self.azimuth = sph[0]
        self.elevation = sph[1]
        

    def angle(self, azimuth, elevation, radians=False):
        '''Get angle between azimuth and elevation and pointing direction.
        
            :param float azimuth: Azimuth east of north to measure from.
            :param float elevation: Elevation from horizon to measure from.
            
            :return: Angle between pointing and given direction.
            :rtype: float
        '''
        direction = coordinates.azel_to_cart(azimuth, elevation, 1.0, radians=radians)
        return coordinates.vector_angle(self.pointing, direction, radians=radians)

    def vector_angle(self, k, radians=False):
        '''Get angle between local direction and pointing direction.
        
            :param numpy.array k: Direction to evaluate angle to.

            :return: Angle between pointing and given direction.
            :rtype: float
        '''
        return coordinates.vector_angle(self.pointing, k, radians=radians)

    @abstractmethod
    def gain(self, k):
        '''Return the gain in the given direction.

        :param numpy.array k: Direction in local coordinates to evaluate gain in.
        
        :return float: Radar gain in the given direction.
        '''
        pass
