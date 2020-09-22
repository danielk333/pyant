#!/usr/bin/env python

from time import ctime
import copy

import numpy as np
import scipy.constants
import scipy.special

from .beam import Beam
from . import coordinates

class PhasedFiniteCylindricalParabola(Beam):
    '''A finite Cylindrical Parabola with a finite receiver line feed in the longitudinal direction, i.e. in the direction of the cylinder.

    :param float I0: Peak gain (linear scale) in the pointing direction.
    :param float width: Panel width in meters
    :param float height: Panel height in meters

    :ivar float I0: Peak gain (linear scale) in the pointing direction.
    :ivar float width: Panel width in meters
    :ivar float height: Panel height in meters
    '''
    def __init__(self, azimuth, elevation, frequency, phase_steering, I0, width, height, depth, **kwargs):
        super().__init__(azimuth, elevation, frequency, **kwargs)
        self.I0 = I0
        self.width = width
        self.height = height
        self.depth = depth
        self.phase_steering = phase_steering

    def copy(self):
        '''Return a copy of the current instance.
        '''
        return PhasedFiniteCylindricalParabola(
            azimuth = copy.deepcopy(self.azimuth),
            elevation = copy.deepcopy(self.elevation),
            frequency = copy.deepcopy(self.frequency),
            phase_steering = copy.deepcopy(self.phase_steering),
            I0 = copy.deepcopy(self.I0),
            width = copy.deepcopy(self.width),
            height = copy.deepcopy(self.height),
            depth = copy.deepcopy(self.depth),
            radians = self.radians,
        )



    def named_shape(self):
        '''Return the named shape of all variables. This can be overridden to extend the possible variables contained in an instance.
        '''
        shape = super().named_shape()
        shape['phase_steering'] = Beam._get_len(self.phase_steering)
        return shape


    def get_parameters(self, ind):
        ind, shape = self.default_ind(ind)
        
        if shape['frequency'] is not None:
            frequency = self.frequency[ind['frequency']]
        else:
            frequency = self.frequency

        if shape['pointing'] is not None:
            pointing = self.pointing[:,ind['pointing']]
        else:
            pointing = self.pointing

        if shape['phase_steering'] is not None:
            phase_steering = self.phase_steering[ind['phase_steering']]
        else:
            phase_steering = self.phase_steering

        return frequency, pointing, phase_steering


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

        theta = np.arcsin(kb[1,...])
        phi = np.arcsin(kb[0,...])

        return theta, phi       # y, x

    def gain(self, k, ind=None, **kw):
        theta, phi = self.local_to_pointing(k, ind) #rad

        return self.gain_tf(theta, phi, **kw)

    def gain_tf(self, theta, phi, ind=None, polarization=None):
        frequency, _, phase_steering = self.get_parameters(ind)
        wavelength = scipy.constants.c/frequency

        if not self.radians:
            phase_steering = np.radians(phase_steering)

        #print(f"phase steering: {phase_steering}")

        print(f" 3 at {ctime()}")

        height = self.height 
        width = self.width - self.depth*np.tan(np.abs(phi)) #depth effective area loss
        # x = longitudinal angle (i.e. parallel to el.axis), 0 = boresight, radians
        # y = transverse angle, 0 = boresight, radians

        x = height/wavelength*np.sin(theta)  # sinc component (longitudinal)
        y = width/wavelength*np.sin(phi - phase_steering)   # sinc component (transverse)
        G = np.sinc(x)*np.sinc(y) # sinc fn. (= field), NB: np.sinc includes pi !!
        G = G*np.cos(phi)         # density (from spherical integration)
        G = G*G                   # sinc^2 fn. (= power)
        # G = np.abs(G)           #amplitude??

        return G*self.I0
