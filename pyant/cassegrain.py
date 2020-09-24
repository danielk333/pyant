#!/usr/bin/env python
import copy

import numpy as np
import scipy.special

from .beam import Beam
from . import coordinates

class Cassegrain(Beam):
    '''Cassegrain gain model of a radar dish.


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

    def copy(self):
        '''Return a copy of the current instance.
        '''
        return Cassegrain(
            azimuth = copy.deepcopy(self.azimuth),
            elevation = copy.deepcopy(self.elevation),
            frequency = copy.deepcopy(self.frequency),
            I0 = copy.deepcopy(self.I0),
            a1 = copy.deepcopy(self.a1),
            a0 = copy.deepcopy(self.a0),
            radians = self.radians,
        )

    def gain(self, k, polarization=None, ind=None):
        pointing, frequency = self.get_parameters(ind)

        theta = coordinates.vector_angle(pointing, k, radians=True)

        lam = scipy.constants.c/frequency
        k_n=2.0*np.pi/lam

        if len(k.shape) == 1:
            theta = np.array([theta], dtype=k.dtype)

        G = np.empty((len(theta),), dtype=k.dtype)
        inds_ = np.pi*np.sin(theta) < 1e-9
        not_inds_ = np.logical_not(inds_)

        G[inds_] = self.I0

        A=(self.I0*((lam/(np.pi*np.sin(theta[not_inds_])))**2.0))/((self.a0**2.0-self.a1**2.0)**2.0)
        B=(self.a0*scipy.special.jn(1,self.a0*np.pi*np.sin(theta[not_inds_])/lam)-self.a1*scipy.special.jn(1,self.a1*np.pi*np.sin(theta[not_inds_])/lam))**2.0
        A0=(self.I0*((lam/(np.pi*np.sin(1e-6)))**2.0))/((self.a0**2.0-self.a1**2.0)**2.0)
        B0=(self.a0*scipy.special.jn(1,self.a0*np.pi*np.sin(1e-6)/lam)-self.a1*scipy.special.jn(1,self.a1*np.pi*np.sin(1e-6)/lam))**2.0
        const=self.I0/(A0*B0)
        G[not_inds_] = A*B*const

        if len(k.shape) == 1:
            G = G[0]
        
        return G

