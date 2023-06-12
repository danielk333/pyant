#!/usr/bin/env python

import copy

import numpy as np
import scipy.constants
import scipy.special

from ..beam import Beam
from .. import coordinates


class FiniteCylindricalParabola(Beam):
    """A finite Cylindrical Parabola with a finite receiver line feed in the
    longitudinal direction, i.e. in the direction of the cylinder axis.

    Custom (measured or more accurately estimated) peak gain at boresight can
    be input, otherwise assumes width (aperture) and height >> wavelength and
    approximates integral with analytic form.

    Parameters
    ----------
    I0 : float
        Peak gain (linear scale) in the pointing direction. Default use
        approximate analytical integral of 2D Fourier transform of rectangle.
    width : float
        Reflector panel width (axial/azimuth dimension) in meters.
    height : float
        Reflector panel height (perpendicular/elevation dimension) in meters.
    aperture : float
        Optional, Length of the feed in meters.
        Default is same as reflector width.

    """

    def __init__(
        self,
        azimuth,
        elevation,
        frequency,
        width,
        height,
        aperture=None,
        I0=None,
        **kwargs,
    ):
        super().__init__(azimuth, elevation, frequency, **kwargs)
        if aperture is None:
            aperture = width
        self.I0 = I0

        self.register_parameter("width")
        self.register_parameter("height")
        self.register_parameter("aperture")

        self.fill_parameter("width", width)
        self.fill_parameter("height", height)
        self.fill_parameter("aperture", aperture)

    def normalize(self, width, height, wavelength):
        """Calculate normalization constant for beam pattern by assuming
        width and height >> wavelength.
        """
        return 4 * np.pi * width * height / wavelength**2

    @property
    def width(self):
        """Reflector panel width (axial/azimuth dimension) in meters."""
        return self.parameters["width"]

    @width.setter
    def width(self, val):
        self.fill_parameter("width", val)

    @property
    def height(self):
        """Reflector panel height (perpendicular/elevation dimension) in meters."""
        return self.parameters["height"]

    @height.setter
    def height(self, val):
        self.fill_parameter("height", val)

    @property
    def aperture(self):
        """Length of the feed in meters."""
        return self.parameters["aperture"]

    @aperture.setter
    def aperture(self, val):
        self.fill_parameter("aperture", val)

    def copy(self):
        """Return a copy of the current instance."""
        return FiniteCylindricalParabola(
            azimuth=copy.deepcopy(self.azimuth),
            elevation=copy.deepcopy(self.elevation),
            frequency=copy.deepcopy(self.frequency),
            I0=copy.deepcopy(self.I0),
            width=copy.deepcopy(self.width),
            height=copy.deepcopy(self.height),
            aperture=copy.deepcopy(self.aperture),
            degrees=self.degrees,
        )

    def local_to_pointing(self, k, azimuth, elevation, degrees=None):
        """Convert from local wave vector direction to bore-sight relative
        longitudinal and transverse angles.
        """
        if degrees is None:
            degrees = self.degrees

        k_ = k / np.linalg.norm(k, axis=0)

        if degrees:
            ang_ = 90.0
        else:
            ang_ = np.pi / 2

        Rz = coordinates.rot_mat_z(azimuth, degrees=degrees)
        Rx = coordinates.rot_mat_x(ang_ - elevation, degrees=degrees)

        # Look direction rotated into the radar's boresight system
        kb = Rx @ Rz @ k_

        # angle of kb from x;z plane, counter-clock wise
        # ( https://www.cv.nrao.edu/~sransom/web/Ch3.html )
        theta = np.arcsin(kb[1, ...])  # Angle of look above (-) or below (+) boresight

        # angle of kb from y;z plane, clock wise
        # ( https://www.cv.nrao.edu/~sransom/web/Ch3.html )
        phi = np.arcsin(kb[0, ...])  # Angle of look to left (-) or right (+) of b.s.

        if degrees:
            theta = np.degrees(theta)
            phi = np.degrees(phi)

        return theta, phi

    def pointing_to_local(self, theta, phi, azimuth, elevation, degrees=None):
        """Convert from bore-sight relative longitudinal and transverse angles
        to local wave vector direction.
        """
        if degrees is None:
            degrees = self.degrees

        sz = (3,)
        if isinstance(theta, np.ndarray):
            sz = sz + (len(theta),)
        elif isinstance(phi, np.ndarray):
            sz = sz + (len(phi),)

        if degrees:
            theta = np.radians(theta)
            phi = np.radians(phi)

        kb = np.zeros(sz, dtype=np.float64)
        kb[1, ...] = np.sin(theta)
        kb[0, ...] = np.sin(phi)
        kb[2, ...] = np.sqrt(1 - kb[0, ...] ** 2 - kb[1, ...] ** 2)

        if degrees:
            ang_ = 90.0
        else:
            ang_ = np.pi / 2

        Rz = coordinates.rot_mat_z(azimuth, degrees=degrees)
        Rx = coordinates.rot_mat_x(ang_ - elevation, degrees=degrees)

        # Look direction rotated from the radar's boresight system
        k = Rz.T @ Rx.T @ kb

        return k

    def gain(self, k, ind=None, polarization=None, **kwargs):
        assert len(k.shape) <= 2, "'k' can only be vectorized with one additional axis"

        params, shape = self.get_parameters(ind, named=True, max_vectors=0)
        if len(params["pointing"].shape) == 2:
            params["pointing"] = params["pointing"].reshape(3)

        sph = coordinates.cart_to_sph(params["pointing"], degrees=False)
        theta, phi = self.local_to_pointing(k, sph[0], sph[1], degrees=False)

        return self.gain_tf(theta, phi, params, degrees=False)

    def gain_tf(self, theta, phi, params, degrees=None):
        """Calculate gain in the frame rotated to the aperture plane.

        Parameters
        ----------
        theta : numpy.ndarray
            The below-axis angle. When elevation < 90, positive theta tends
            towards the horizon, and negative theta towards zenith.
        phi : numpy.ndarray
            The off-axis angle. When looking out along boresight with the
            azimuth direction straight ahead, positive phi is to your right,
            negative phi to your left.
        params : dict
            The parameters to use for gain calculation.
        """
        if degrees is None:
            degrees = self.degrees

        wavelength = scipy.constants.c / params["frequency"]

        if self.I0 is None:
            I0 = self.normalize(params["aperture"], params["height"], wavelength)
        else:
            I0 = self.I0

        if degrees:
            theta = np.radians(theta)
            phi = np.radians(phi)

        # x = longitudinal angle (i.e. parallel to el.axis), 0 = boresight, radians
        # y = transverse angle, 0 = boresight, radians

        # sinc*sinc is 2D FFT of a rectangular aperture
        x = params["aperture"] / wavelength * np.sin(phi)  # sinc component (longitudinal)
        y = params["height"] / wavelength * np.sin(theta)  # sinc component (transverse)
        G = np.sinc(x) * np.sinc(y)  # sinc fn. (= field), NB: np.sinc includes pi !!
        G *= np.cos(phi)  # Element gain
        G = G * G  # sinc^2 fn. (= power)

        return G * I0
