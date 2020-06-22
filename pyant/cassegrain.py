#!/usr/bin/env python

import numpy as np
import scipy.special

from .beam import Beam
from . import coordinates

class Cassegrain(Beam):
    '''
    :param float I0: Peak gain (linear scale) in the pointing direction.
    :param float a0: Radius longitudinal direction
    :param float a1: Radius latitudinal direction

    :ivar float I0: Peak gain (linear scale) in the pointing direction.
    :ivar float a0: Radius longitudinal direction
    :ivar float a1: Radius latitudinal direction
    '''
    def __init__(self, azimuth, elevation, frequency, I0, a0, a1, **kwargs):
        super().__init__(azimuth, elevation, frequency, **kwargs)
        self.I0 = I0
        self.a1 = a1
        self.a0 = a0


    def gain(self, k):
        theta = coordinates.vector_angle(self.pointing, k, radians=True)

        if theta < 1e-6:
            return self.I0

        lam = self.wavelength
        k_n=2.0*np.pi/lam

        A=(self.I0*((lam/(np.pi*np.sin(theta)))**2.0))/((self.a0**2.0-self.a1**2.0)**2.0)
        B=(self.a0*scipy.special.jn(1,self.a0*np.pi*np.sin(theta)/lam)-self.a1*scipy.special.jn(1,self.a1*np.pi*np.sin(theta)/lam))**2.0
        A0=(self.I0*((lam/(np.pi*np.sin(1e-6)))**2.0))/((self.a0**2.0-self.a1**2.0)**2.0)
        B0=(self.a0*scipy.special.jn(1,self.a0*np.pi*np.sin(1e-6)/lam)-self.a1*scipy.special.jn(1,self.a1*np.pi*np.sin(1e-6)/lam))**2.0
        const=self.I0/(A0*B0)

        return A*B*const

