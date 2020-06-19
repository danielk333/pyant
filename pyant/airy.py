#!/usr/bin/env python

import numpy as np
import scipy.constants
import scipy.special

from .beam import Beam
from . import coordinates

class Airy(Beam):
    '''
    :param float I0: Peak gain (linear scale) in the pointing direction.
    :param float radius: Radius in meters of the airy disk

    :ivar float I0: Peak gain (linear scale) in the pointing direction.
    :ivar float radius: Radius in meters of the airy disk
    '''
    def __init__(self, azimuth, elevation, frequency, I0, radius, **kwargs):
        super().__init__(azimuth, elevation, frequency, **kwargs)
        self.I0 = I0
        self.radius = radius


    def gain(self, k):
        theta = coordinates.vector_angle(self.pointing, k, radians=True)

        if theta < 1e-6:
            return self.I0

        lam = scipy.constants.c/self.frequency
        k_n = 2.0*np.pi/lam
        alph = k_n*self.radius*np.sin(theta)
        jn_val = scipy.special.jn(1,alph)
        G = self.I0*((2.0*jn_val/alph))**2.0

        return G

