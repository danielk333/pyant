#!/usr/bin/env python

from time import ctime
import copy

import numpy as np
import scipy.constants
import scipy.special

from .beam import Beam
from . import coordinates

class FiniteCylindricalParabola(Beam):
    """
    A finite Cylindrical Parabola with a finite receiver line feed
        in the longitudinal direction, i.e. in the direction of the cylinder axis.

    Custom (measured or more accurately estimated) peak gain at boresight can
    be input, otherwise assumes width (aperture) and height >> wavelength and approximates
    integral with analytic form.

    :param float I0: Peak gain (linear scale) in the pointing direction.
                    Default use approximate analytical integral of 2D Fourier transform of rectangle.
    :param float width:  Reflector panel width (axial/azimuth dimension) in meters
    :param float height: Reflector panel height (perpendicular/elevation dimension) in meters
    :param float aperture: Optional, Length of the feed in meters.  Default is same as reflector width
    :param float rotation: DISABLED Optional, Rotation of the rectangle in the local coordinate system.
                    If no rotation angle is given, the width is along the `y` (north-south) axis in local coordinates.

    """
    def __init__(self, azimuth, elevation, frequency, width, height, aperture=None, I0=None, rotation=None, **kwargs):
        super().__init__(azimuth, elevation, frequency, **kwargs)
        self.width = width
        self.height = height
        if aperture is None:
            aperture = width
        self.aperture = aperture
        self.I0 = I0
        self.rotation = rotation

    def normalize(self, width, height, wavelength):
        '''Calculate normalization constant for beam pattern by assuming width and height >> wavelength.
        '''
        return 4*np.pi*width*height/wavelength**2


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
            aperture = copy.deepcopy(self.aperture),
            radians = self.radians,
        )

    def local_to_pointing(self, k, azimuth, elevation):
        '''Convert from local wave vector direction to bore-sight relative longitudinal and transverse angles.
        '''
        k_ = k/np.linalg.norm(k, axis=0)

        if self.radians:
            ang_ = np.pi/2
        else:
            ang_ = 90.0

        def Rz(azimuth):
            return coordinates.rot_mat_z(azimuth, radians=self.radians)
        def Rx(elevation):
            return coordinates.rot_mat_x(ang_ - elevation, radians=self.radians)

        # Look direction rotated into the radar's boresight system
        kb = Rx(elevation) @ Rz(azimuth) @ k_

        #if the rectangular aperture is rotated, apply a rotation
        # DISABLED (TG) -- I don't know what this is supposed to achieve
        # which isn't already covered by `azimuth`.  But since it is applied
        # _after_ the `azimuth` and `elevation` rotations, the semantics are
        # not clear to me.
        if self.rotation is not None:
            # raise NotImplementedError('Try using `azimuth` instead')
            Rz_ant = coordinates.rot_mat_z(-self.rotation, radians = self.radians)
            kb = Rz_ant.dot(kb)

        #angle of kb from x;z plane, counter-clock wise ( https://www.cv.nrao.edu/~sransom/web/Ch3.html )
        theta = np.arcsin(kb[1,...])    # Angle of look above (-) or below (+) boresight

        #angle of kb from y;z plane, clock wise ( https://www.cv.nrao.edu/~sransom/web/Ch3.html )
        phi = np.arcsin(kb[0,...])      # Angle of look to left (-) or right (+) of b.s.

        return theta, phi



    def pointing_to_local(self, theta, phi, azimuth, elevation):
        '''Convert from bore-sight relative longitudinal and transverse angles to local wave vector direction.
        '''
        sz = (3,)
        if isinstance(theta, np.ndarray):
            sz = sz + (len(theta),)
        elif isinstance(phi, np.ndarray):
            sz = sz + (len(phi),)

        if not self.radians:
            theta = np.degrees(theta)
            phi = np.degrees(phi)

        kb = np.zeros(sz, dtype=np.float64)
        kb[1,...] = np.sin(theta)
        kb[0,...] = np.sin(phi)
        kb[2,...] = np.sqrt(1 - kb[0,...]**2 - kb[1,...]**2)

        if self.radians:
            ang_ = np.pi/2
        else:
            ang_ = 90.0

        def Rz(azimuth):
            return coordinates.rot_mat_z(azimuth, radians=self.radians)
        def Rx(elevation):
            return coordinates.rot_mat_x(ang_ - elevation, radians=self.radians)

        # Look direction rotated from the radar's boresight system
        k =  Rz(azimuth).T @ Rx(elevation).T @ kb

        return k





    def gain(self, k, ind=None, polarization=None, **kwargs):
        pointing, frequency = self.get_parameters(ind, **kwargs)

        sph = coordinates.cart_to_sph(pointing, radians = self.radians)

        theta, phi = self.local_to_pointing(k, sph[0], sph[1])

        return self.gain_tf(theta, phi, frequency=frequency)

    def gain_tf(self, theta, phi, ind=None, **kwargs):
        """
        theta is below-axis angle.
        When elevation < 90, positive theta tends towards the horizon,
        and negative theta towards zenith.

        phi is off-axis angle.
        When looking out along boresight with the azimuth direction straight
        ahead, positive phi is to your right, negative phi to your left.
        """
        if 'frequency' not in kwargs:
            _, frequency = self.get_parameters(ind, **kwargs)
        else:
            frequency = kwargs['frequency']

        wavelength = scipy.constants.c/frequency

        if self.I0 is None:
            I0 = self.normalize(self.aperture, self.height, wavelength)
        else:
            I0 = self.I0

        # x = longitudinal angle (i.e. parallel to el.axis), 0 = boresight, radians
        # y = transverse angle, 0 = boresight, radians

        #sinc*sinc is 2D FFT of a rectangular aperture
        x = self.aperture/wavelength*np.sin(phi)      # sinc component (longitudinal)
        y = self.height/wavelength*np.sin(theta)      # sinc component (transverse)
        G = np.sinc(x)*np.sinc(y) # sinc fn. (= field), NB: np.sinc includes pi !!
        G *= np.cos(phi)          # Element gain
        G = G*G                   # sinc^2 fn. (= power)

        return G*I0
