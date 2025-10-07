#!/usr/bin/env python

import copy

import numpy as np
from numpy.typing import NDArray
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
        pointing,
        frequency,
        width,
        height,
        aperture_width,
        peak_gain=None,
    ):
        super().__init__()
        self.parameters["pointing"] = pointing
        self.parameters_shape["pointing"] = (3,)
        self.parameters["frequency"] = frequency
        self.parameters["height"] = height
        self.parameters["width"] = width
        self.parameters["aperture_width"] = aperture_width

        self.peak_gain = peak_gain
        self.validate_parameter_shapes()

    def normalize(self, width, height, wavelength):
        """Calculate normalization constant for beam pattern by assuming
        width and height >> wavelength.
        """
        return 4 * np.pi * width * height / wavelength**2

    def copy(self):
        """Return a copy of the current instance."""
        return FiniteCylindricalParabola(
            pointing=copy.deepcopy(self.parameters["pointing"]),
            frequency=copy.deepcopy(self.parameters["frequency"]),
            height=copy.deepcopy(self.parameters["height"]),
            width=copy.deepcopy(self.parameters["width"]),
            aperture_width=copy.deepcopy(self.parameters["aperture_width"]),
            peak_gain=self.peak_gain,
        )

    def local_to_pointing(self, k):
        """Convert from local wave vector direction to bore-sight relative
        longitudinal and transverse angles.
        """
        k_len = self.validate_k_shape(k)
        azelr = coordinates.cart_to_sph(self.parameters["pointing"], degrees=False)
        size = self.size

        k = k / np.linalg.norm(k, axis=0)

        Rz = coordinates.rot_mat_z(azelr[0, ...], degrees=False)
        Rx = coordinates.rot_mat_x(np.pi / 2 - azelr[1, ...], degrees=False)

        # Look direction rotated into the radar's boresight system
        if size > 0 and k_len > 0:
            kb = np.einsum("ijk,jk->ik", Rx, np.einsum("ijk,jk->ik", Rz, k))
        elif size > 0 and k_len == 0:
            kb = np.einsum(
                "ijk,jk->ik",
                Rx,
                np.einsum("ijk,jk->ik", Rz, np.broadcast_to(k.reshape(3, 1), (3, size))),
            )
        elif size == 0:
            kb = Rx @ Rz @ k

        # angle of kb from y;z plane, clock wise
        # ( https://www.cv.nrao.edu/~sransom/web/Ch3.html )
        phi = np.arcsin(kb[0, ...])  # Angle of look to left (-) or right (+) of b.s.

        # angle of kb from x;z plane, counter-clock wise
        # ( https://www.cv.nrao.edu/~sransom/web/Ch3.html )
        theta = np.arcsin(kb[1, ...])  # Angle of look above (-) or below (+) boresight

        return theta, phi

    def pointing_to_local(self, theta, phi):
        """Convert from bore-sight relative longitudinal and transverse angles
        to local wave vector direction.
        """
        azelr = coordinates.cart_to_sph(self.parameters["pointing"], degrees=False)
        size = self.size

        sz = (3,)
        k_len = 0
        if isinstance(theta, np.ndarray) and theta.ndim > 0:
            sz = sz + (len(theta),)
            k_len = len(theta)
        elif isinstance(phi, np.ndarray) and phi.ndim > 0:
            sz = sz + (len(phi),)
            k_len = len(phi)

        kb = np.zeros(sz, dtype=np.float64)
        kb[0, ...] = np.sin(phi)
        kb[1, ...] = np.sin(theta)
        kb[2, ...] = np.sqrt(1 - kb[0, ...] ** 2 - kb[1, ...] ** 2)

        Rz = coordinates.rot_mat_z(azelr[0, ...], degrees=False)
        Rx = coordinates.rot_mat_x(np.pi / 2 - azelr[1, ...], degrees=False)

        # Look direction rotated from the radar's boresight system
        if size > 0 and k_len > 0:
            k = np.einsum(
                "ijk,jk->ik",
                np.einsum("ijk->jik", Rx),
                np.einsum("ijk,jk->ik", np.einsum("ijk->jik", Rz), kb),
            )
        elif size > 0 and k_len == 0:
            k = np.einsum(
                "ijk,jk->ik",
                np.einsum("ijk->jik", Rx),
                np.einsum(
                    "ijk,jk->ik",
                    np.einsum("ijk->jik", Rz),
                    np.broadcast_to(kb.reshape(3, 1), (3, size)),
                ),
            )
        elif size == 0:
            k = Rz.T @ Rx.T @ kb

        return k

    def gain(self, k: NDArray, polarization: NDArray | None = None):
        theta, phi = self.local_to_pointing(k)
        return self.gain_tf(theta, phi)

    def gain_tf(self, theta, phi):
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
        wavelength = scipy.constants.c / self.parameters["frequency"]

        if self.peak_gain is None:
            g0 = self.normalize(
                self.parameters["aperture_width"], self.parameters["height"], wavelength
            )
        else:
            g0 = self.peak_gain

        # x = longitudinal angle (i.e. parallel to el.axis), 0 = boresight, radians
        # y = transverse angle, 0 = boresight, radians

        # sinc*sinc is 2D FFT of a rectangular aperture

        # sinc component (longitudinal)
        x = self.parameters["aperture_width"] / wavelength * np.sin(phi)

        # sinc component (transverse)
        y = self.parameters["height"] / wavelength * np.sin(theta)

        g = np.sinc(x) * np.sinc(y)  # sinc fn. (= field), NB: np.sinc includes pi !!
        g *= np.cos(phi)  # Element gain
        g = g * g  # sinc^2 fn. (= power)

        return g * g0
