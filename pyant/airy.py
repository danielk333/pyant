#!/usr/bin/env python

import numpy as np
import scipy.constants
import scipy.special

from .beam import Beam
from . import coordinates

class Airy(Beam):
    '''Airy disk gain model of a radar dish.


    :param float I0: Peak gain (linear scale) in the pointing direction.
    :param float radius: Radius in meters of the airy disk

    :ivar float I0: Peak gain (linear scale) in the pointing direction.
    :ivar float radius: Radius in meters of the airy disk

    **Notes:**

    * To avoid singularity at beam center, use :math:`\\lim_{x\\mapsto 0} \\frac{J_1(x)}{x} = \\frac{1}{2}` and a threshold.

    '''
    def __init__(self, azimuth, elevation, frequency, I0, radius, **kwargs):
        super().__init__(azimuth, elevation, frequency, **kwargs)
        self.I0 = I0
        self.radius = radius


    def gain(self, k, polarization=None, ind=None):
        ind, shape = self.default_ind(ind)

        if shape['frequency'] is not None:
            lam = self.wavelength[ind['frequency']]
        else:
            lam = self.wavelength

        if shape['pointing'] is not None:
            point = self.pointing[:,ind['pointing']]
        else:
            point = self.pointing
        
        theta = coordinates.vector_angle(point, k, radians=True)
        
        k_n = 2.0*np.pi/lam
        alph = k_n*self.radius*np.sin(theta)
        jn_val = scipy.special.jn(1,alph)

        if len(k.shape) == 1:
            #lim_(alph->0) (J_1(alph))/alph = 1/2
            if alph < 1e-9:
                G = self.I0
            else:
                G = self.I0*((2.0*jn_val/alph))**2.0
        else:
            G = np.empty((k.shape[1],), dtype=k.dtype)
            inds_ = alph < 1e-9
            not_inds_ = np.logical_not(inds_)
            G[inds_] = self.I0
            G[not_inds_] = self.I0*((2.0*jn_val[not_inds_]/alph[not_inds_]))**2.0

        return G

