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
    '''A finite Cylindrical Parabola with a phased finite receiver line feed
        in the longitudinal direction, i.e. in the direction of the cylinder axis.

    Custom (measured or more accurately estimated) peak gain at boresight can
    be input, otherwise assumes width and height >> wavelength and approximates
    integral with analytic form.

    :param float I0: Peak gain (linear scale) in the pointing direction.
    :param float width:  Reflector width (axial/azimuth dimension) in meters
    :param float height: Reflector height (perpendicular/elevation dimension) in meters
    :param float depth: Perpendicular distance from feed to reflector in meters
    :param float aperture: Optional, Length of the feed in meters.  Default is same as reflector width

    :param float I0: Peak gain (linear scale) in the pointing direction.
                    Default use approximate analytical integral of 2D Fourier transform of rectangle.
    :param float rotation: DISABLED Optional, Rotation of the rectangle in the local coordinate system.
                    If no rotation angle is given, the width is along the `y` (north-south) axis in local coordinates.
    '''
    def __init__(self, azimuth, elevation, frequency, phase_steering, width, height, depth, aperture=None, I0=None, rotation=None, **kwargs):
        super().__init__(azimuth, elevation, frequency, width, height, aperture, I0=I0, rotation=rotation, **kwargs)
        self.depth = depth
        self.phase_steering = phase_steering
        self.depth = depth

        self.register_parameter('phase_steering')

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
            aperture = copy.deepcopy(self.aperture),
            I0 = copy.deepcopy(self.I0),
            rotation = copy.deepcopy(self.rotation),
            radians = self.radians,
        )

    def gain(self, k, ind=None, polarization=None, **kwargs):
        params = self.get_parameters(ind, named=True, **kwargs)

        sph = coordinates.cart_to_sph(params['pointing'], radians = self.radians)

        theta, phi = self.local_to_pointing(k, sph[0], sph[1])

        return self.gain_tf(theta, phi, called_from_gain = True, **params)

    # interface method `gain()`, inherited from super, defers to `gain_tf(), below`
    def gain_tf(self, theta, phi, ind=None, **kwargs):
        """
        theta is below-axis angle (radians).
        When elevation < 90, positive theta tends towards the horizon,
        and negative theta towards zenith.

        phi is off-axis angle (radians).
        When looking out along boresight with the azimuth direction straight
        ahead, positive phi is to your right, negative phi to your left.
        """

        #small efficiency fix to skip unnecessary calls of get_parameters
        if 'called_from_gain' in kwargs:
            frequency = kwargs['frequency']
            phase_steering = kwargs['phase_steering']
        else:
            _, frequency, phase_steering = self.get_parameters(ind, **kwargs)

        wavelength = scipy.constants.c/frequency

        if not self.radians:
            phase_steering = np.radians(phase_steering)


        # Compute effective area loss due to spillover when phase-steering past edge of reflector.
        w_loss = np.clip(self.depth*np.tan(np.abs(phase_steering)) - (self.width - self.aperture)/2, 0, None)

        # This implies geometric optics for the feed-to-reflector path
        w_eff = self.aperture - w_loss
        height = self.height

        if self.I0 is None:
            I0 = self.normalize(w_eff, height, wavelength)
        else:
            I0 = self.I0

        # y = longitudinal angle (i.e. parallel to el.axis), 0 = boresight, radians
        # x = transverse angle, 0 = boresight, radians
        x = w_eff/wavelength*(np.sin(phi) - phase_steering)  # sinc component (transverse)
        y = height/wavelength*np.sin(theta)   # sinc component (longitudinal)
        G = np.sinc(x)*np.sinc(y) # sinc fn. (= field), NB: np.sinc includes pi !!

        # NB! This factor of cos(phi) is NOT geometric foreshortening --
        # It is instead a crude model for element (i.e., dipole) gain!
        G *= np.cos(phi)

        G = G*G                   # sinc^2 fn. (= power)

        return G*I0
