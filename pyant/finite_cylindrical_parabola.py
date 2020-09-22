#!/usr/bin/env python
import copy

import numpy as np
import scipy.constants
import scipy.special

from .beam import Beam
from . import coordinates

class FiniteCylindricalParabola(Beam):
    '''A finite Cylindrical Parabola with a finite receiver line feed in the longitudinal direction, i.e. in the direction of the cylinder.
    Custom peak gain can be input, otherwise assumes width and height >> wavelength and approximates integral with analytic form.

    :param float I0: Peak gain (linear scale) in the pointing direction. Default use approximate analytical integral of 2D Fourier transform of rectangle.
    :param float width: Panel width in meters
    :param float height: Panel height in meters
    :param float rotation: Rotation of the rectangle in the local coordinate system. If no rotation angle is given, the width is along the `y` (north-south) axis in local coordinates.

    :ivar float I0: Peak gain (linear scale) in the pointing direction.
    :ivar float width: Panel width in meters
    :ivar float height: Panel height in meters
    :ivar float rotation: Rotation of the rectangle in the local coordinate system. If no rotation angle is given, the width is along the `y` (north-south) axis in local coordinates.
    '''
    def __init__(self, azimuth, elevation, frequency, width, height, I0=None, rotation=None, **kwargs):
        super().__init__(azimuth, elevation, frequency, **kwargs)
        self.I0 = I0
        self.width = width
        self.height = height
        self.rotation = rotation

        if self.I0 is None:
            self.I0 = self.normalize()

    def normalize(self):
        '''Calculate normalization constant for beam pattern by assuming width and height >> wavelength.
        '''
        return 4*np.pi*self.width*self.height/self.wavelength**2


    def copy(self):
        '''Return a copy of the current instance.
        '''
        return FiniteCylindricalParabola(
            azimuth = copy.deepcopy(self.azimuth),
            elevation = copy.deepcopy(self.elevation),
            frequency = copy.deepcopy(self.frequency),
            I0 = copy.deepcopy(self.I0),
            width = copy.deepcopy(self.width),
            height = copy.deepcopy(self.height),
            radians = self.radians,
        )

    def local_to_pointing(self, k, ind=None):
        '''Convert from local wave vector direction to bore-sight relative longitudinal and transverse angles.
        '''
        k_ = k/np.linalg.norm(k, axis=0)

        ind, shape = self.default_ind(ind)
        if shape['pointing'] is not None:
            azimuth = self.azimuth[ind['pointing']]
            elevation = self.elevation[ind['pointing']]
        else:
            azimuth = self.azimuth
            elevation = self.elevation


        if self.radians:
            ang_ = np.pi/2
        else:
            ang_ = 90.0

        Rz = coordinates.rot_mat_z(azimuth, radians = self.radians)
        Rx = coordinates.rot_mat_x(ang_ - elevation, radians = self.radians)

        kb = Rx.dot(Rz.dot(k_))
        if self.rotation is not None:
            Rz_ant = coordinates.rot_mat_z(self.rotation, radians = self.radians)
            kb = Rz_ant.dot(kb)

        theta = np.arcsin(kb[1,...]) #angle from x;z plane, counter-clock wise
        phi = -np.arcsin(kb[0,...]) #angle from y;z plane, counter-clock wise

        return theta, phi

    def gain(self, k, polarization=None, ind=None):
        frequency, _ = self.get_parameters(ind)
        theta, phi = self.local_to_pointing(k, ind)

        wavelength = scipy.constants.c/frequency

        # x = longitudinal angle (i.e. parallel to el.axis), 0 = boresight, radians
        # y = transverse angle, 0 = boresight, radians
        x = self.width/wavelength*np.sin(theta)    # sinc component (longitudinal)
        y = self.height/wavelength*np.sin(phi)      # sinc component (transverse)
        G = np.sinc(x)*np.sinc(y) # sinc fn. (= field), NB: np.sinc includes pi !!
        G = G*G                   # sinc^2 fn. (= power)

        return G*self.I0
