#!/usr/bin/env python

import numpy as np
import scipy.constants
import scipy.special

from .beam import Beam
from . import coordinates

class Gaussian(Beam):
    '''Gaussian tapered planar array model

    :param float I0: Peak gain (linear scale) in the pointing direction.
    :param float radius: Radius in meters of the planar array
    :param float normal_azimuth: Azimuth of normal vector of the planar array in dgreees.
    :param float normal_elevation: Elevation of pointing direction in degrees.

    :ivar float I0: Peak gain (linear scale) in the pointing direction.
    :ivar float radius: Radius in meters of the airy disk
    :ivar numpy.ndarray normal: Planar array normal vector in local coordinates
    :ivar float normal_azimuth: Azimuth of normal vector of the planar array in dgreees.
    :ivar float normal_elevation: Elevation of pointing direction in degrees.
    '''
    def __init__(self, azimuth, elevation, frequency, I0, radius, normal_azimuth, normal_elevation, **kwargs):
        super().__init__(azimuth, elevation, frequency, **kwargs)
        self.I0 = I0
        self.radius = radius
        self.normal_azimuth = normal_azimuth
        self.normal_elevation = normal_elevation
        self.normal = coordinates.sph_to_cart(
            np.array([normal_azimuth, normal_elevation, 1]),
            radians = self.radians,
        )


    def gain(self, k, polarization=None, ind=None):
        frequency, pointing = self.get_parameters(ind)
        lam = scipy.constants.c/frequency

        if np.abs(1-np.dot(pointing,self.normal)) < 1e-6:
            rd=np.random.randn(3)
            rd=rd/np.sqrt(np.dot(rd,rd))
            ct=np.cross(self.pointing,rd)
        else:
            ct=np.cross(self.pointing,self.normal)
        
        ct=ct/np.sqrt(np.dot(ct,ct))
        ht=np.cross(self.normal,ct)
        ht=ht/np.sqrt(np.dot(ht,ht))
        angle = coordinates.vector_angle(pointing, ht, radians=True)

        ot=np.cross(pointing,ct)
        ot=ot/np.sqrt(np.dot(ot,ot))

        I_1=np.sin(angle)*self.I0
        a0p=np.sin(angle)*self.radius

        sigma1=0.7*a0p/lam
        sigma2=0.7*self.radius/lam

        k0=k/np.linalg.norm(k,axis=0)
        
        l1=np.dot(ct,k0)
        m1=np.dot(ot,k0)
        
        l2=l1*l1
        m2=m1*m1
        G = I_1*np.exp(-np.pi*m2*2.0*np.pi*sigma1**2.0)*np.exp(-np.pi*l2*2.0*np.pi*sigma2**2.0)
        return G


