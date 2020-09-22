#!/usr/bin/env python

from time import ctime
import copy

import numpy as np
import scipy.constants
import scipy.special

from .beam import Beam
from . import coordinates
from .finite_cylindrical_parabola import FiniteCylindricalParabola

class PhasedFiniteCylindricalParabola(FiniteCylindricalParabola):
    '''A finite Cylindrical Parabola with a finite receiver line feed in the longitudinal direction, i.e. in the direction of the cylinder.

    :param float I0: Peak gain (linear scale) in the pointing direction.
    :param float width: Panel width in meters
    :param float height: Panel height in meters

    :ivar float I0: Peak gain (linear scale) in the pointing direction.
    :ivar float width: Panel width in meters
    :ivar float height: Panel height in meters
    '''
    def __init__(self, azimuth, elevation, frequency, phase_steering, width, height, depth, I0=None, rotation=None, **kwargs):
        super().__init__(azimuth, elevation, frequency, width, height, I0=I0, rotation=rotation, **kwargs)
        self.depth = depth
        self.phase_steering = phase_steering

        self.register_parameter('phase_steering', axis=0, default_ind=0)

    def copy(self):
        '''Return a copy of the current instance.
        '''
        return PhasedFiniteCylindricalParabola(
            azimuth = copy.deepcopy(self.azimuth),
            elevation = copy.deepcopy(self.elevation),
            frequency = copy.deepcopy(self.frequency),
            phase_steering = copy.deepcopy(self.phase_steering),
            width = copy.deepcopy(self.width),
            height = copy.deepcopy(self.height),
            depth = copy.deepcopy(self.depth),
            I0 = copy.deepcopy(self.I0),
            rotation = copy.deepcopy(self.rotation),
            radians = self.radians,
        )


    def gain(self, k, polarization=None, ind=None):
        theta, phi = self.local_to_pointing(k, ind) #rad

        return self.gain_tf(theta, phi, polarization=polarization, ind=ind)

    def gain_tf(self, theta, phi, polarization=None, ind=None):
        _, frequency, phase_steering = self.get_parameters(ind)
        wavelength = scipy.constants.c/frequency

        if not self.radians:
            phase_steering = np.radians(phase_steering)

        if self.I0 is None:
            I0 = self.normalize(wavelength)
        else:
            I0 = self.I0

        height = self.height 
        width = self.width - self.depth*np.tan(np.abs(phi)) #depth effective area loss
        # x = longitudinal angle (i.e. parallel to el.axis), 0 = boresight, radians
        # y = transverse angle, 0 = boresight, radians

        x = width/wavelength*np.sin(phi - phase_steering)  # sinc component (longitudinal)
        y = height/wavelength*np.sin(theta)   # sinc component (transverse)
        G = np.sinc(x)*np.sinc(y) # sinc fn. (= field), NB: np.sinc includes pi !!
        G = G*G                   # sinc^2 fn. (= power)

        return G*I0
