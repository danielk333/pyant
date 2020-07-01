#!/usr/bin/env python

import numpy as np
import scipy.constants
import scipy.special

from .beam import Beam
from . import coordinates

class FiniteCylindricalParabola(Beam):
    '''A finite Cylindrical Parabola with a finite receiver line feed in the longitudinal direction, i.e. in the direction of the cylinder.

    :param float I0: Peak gain (linear scale) in the pointing direction.
    :param float width: Panel width in meters
    :param float height: Panel height in meters

    :ivar float I0: Peak gain (linear scale) in the pointing direction.
    :ivar float width: Panel width in meters
    :ivar float height: Panel height in meters
    '''
    def __init__(self, azimuth, elevation, frequency, I0, width, height, **kwargs):
        super().__init__(azimuth, elevation, frequency, **kwargs)
        self.I0 = I0
        self.width = width
        self.height = height

    def local_to_pointing(self, k):
        '''Convert from local wave vector direction to bore-sight relative longitudinal and transverse angles.
        '''
        k_ = k/np.linalg.norm(k, axis=0)
        
        if self.radians:
            ang_ = np.pi/2
        else:
            ang_ = 90.0

        Rz = coordinates.rot_mat_z(self.azimuth, radians = self.radians)
        Rx = coordinates.rot_mat_x(ang_ - self.elevation, radians = self.radians)

        kb = Rx.dot(Rz.dot(k_))

        theta = np.arcsin(kb[1,...])
        phi = np.arcsin(kb[0,...])

        return theta, phi

    def gain(self, k, polarization=None, ind=None):
        theta, phi = self.local_to_pointing(k)

        # x = longitudinal angle (i.e. parallel to el.axis), 0 = boresight, radians
        # y = transverse angle, 0 = boresight, radians
        x = self.width/self.wavelength*np.sin(theta)    # sinc component (longitudinal)
        y = self.height/self.wavelength*np.sin(phi)      # sinc component (transverse)
        G = np.sinc(x)*np.sinc(y) # sinc fn. (= field), NB: np.sinc includes pi !!
        G = G*np.cos(phi)         # density (from spherical integration)
        G = G*G                   # sinc^2 fn. (= power)
        # G = np.abs(G) #amplitude??

        return G*self.I0
