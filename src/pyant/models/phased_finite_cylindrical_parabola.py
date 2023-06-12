#!/usr/bin/env python

import copy

import numpy as np
import scipy.constants
import scipy.special

from .finite_cylindrical_parabola import FiniteCylindricalParabola


class PhasedFiniteCylindricalParabola(FiniteCylindricalParabola):
    """A finite Cylindrical Parabola with a phased finite receiver line feed
        in the longitudinal direction, i.e. in the direction of the cylinder axis.

    Parameters
    ----------
    width : float
        Reflector width (axial/azimuth dimension) in meters
    height : float
        Reflector height (perpendicular/elevation dimension) in meters
    depth : float
        Perpendicular distance from feed to reflector in meters
    phase_steering : float
        Phase steering angle applied to the feed bridge of antennas
    aperture : float
        Optional, Length of the feed in meters.
        Default is same as reflector width
    I0 : float
        Peak gain (linear scale) in the pointing direction.
        Default use approximate analytical integral of 2D Fourier
        transform of rectangle.


    Notes
    ------
    Peak gain
        Custom (measured or more accurately estimated) peak gain at boresight can
        be input, otherwise assumes width and height >> wavelength and approximates
        integral with analytic form.


    """

    def __init__(
        self,
        azimuth,
        elevation,
        frequency,
        phase_steering,
        width,
        height,
        depth,
        aperture=None,
        I0=None,
        **kwargs,
    ):
        super().__init__(
            azimuth, elevation, frequency, width, height, aperture=aperture, I0=I0, **kwargs
        )
        self.register_parameter("depth")
        self.register_parameter("phase_steering")

        self.fill_parameter("depth", depth)
        self.fill_parameter("phase_steering", phase_steering)

    @property
    def depth(self):
        """Perpendicular distance from feed to reflector in meters."""
        return self.parameters["depth"]

    @depth.setter
    def depth(self, val):
        self.fill_parameter("depth", val)

    @property
    def phase_steering(self):
        """Phase steering angle applied to the feed bridge of antennas"""
        return self.parameters["phase_steering"]

    @phase_steering.setter
    def phase_steering(self, val):
        self.fill_parameter("phase_steering", val)

    def copy(self):
        """Return a copy of the current instance."""
        return PhasedFiniteCylindricalParabola(
            azimuth=copy.deepcopy(self.azimuth),
            elevation=copy.deepcopy(self.elevation),
            frequency=copy.deepcopy(self.frequency),
            phase_steering=copy.deepcopy(self.phase_steering),
            width=copy.deepcopy(self.width),
            height=copy.deepcopy(self.height),
            depth=copy.deepcopy(self.depth),
            aperture=copy.deepcopy(self.aperture),
            I0=copy.deepcopy(self.I0),
            radians=self.radians,
        )

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
        if degrees:
            theta = np.radians(theta)
            phi = np.radians(phi)

        if self.degrees:
            phase_steering = np.radians(params["phase_steering"])
        else:
            phase_steering = params["phase_steering"]

        # Compute effective area loss due to spillover when phase-steering past edge of reflector.
        A = (
            params["depth"] * np.tan(np.abs(phase_steering))
            - (params["width"] - params["aperture"]) / 2
        )
        w_loss = np.clip(A, 0, None)

        # This implies geometric optics for the feed-to-reflector path
        w_eff = np.clip(params["aperture"] - w_loss, 0, None)

        if self.I0 is None:
            I0 = self.normalize(w_eff, params["height"], wavelength)
        else:
            I0 = self.I0

        # y = longitudinal angle (i.e. parallel to el.axis), 0 = boresight, radians
        # x = transverse angle, 0 = boresight, radians
        x = w_eff / wavelength * (np.sin(phi) - phase_steering)  # sinc component (transverse)
        y = params["height"] / wavelength * np.sin(theta)  # sinc component (longitudinal)
        G = (
            np.sinc(x) * np.sinc(y) * (w_eff / self.aperture)
        )  # sinc fn. (= field), NB: np.sinc includes pi !!

        # NB! This factor of cos(phi) is NOT geometric foreshortening --
        # It is instead a crude model for element (i.e., dipole) gain!
        G *= np.cos(phi)

        G = G * G  # sinc^2 fn. (= power)

        return G * I0

    def _nominal_phase_steering(self, phi, degrees=None):
        """
        For a given desired pointing `phi`, compute the nominal phase
        steering angle that gives a pattern that peaks at `phi`.

        We know the azimuthal gain pattern `G(phi)` for a nominal phase
        steering angle `phi0`; it is given by

            G(phi) = sinc(w_eff / lambda * (sin(phi) - phi0))

        So the pattern has a peak when the argument to `sinc` is zero, ie., at

            phi0 = sin(phi)

        or

            phi = arcsin(phi0)

        This also means that steering past phi = 1, or 57.3°, is not possible.

        """
        if degrees is None:
            degrees = self.degrees
        if degrees:
            phi = np.radians(phi)
        assert np.all(np.abs(phi) <= np.pi / 2), f"cannot steer past the horizon {np.abs(phi)}"

        phi0 = np.sin(phi)

        if degrees:
            phi0 = np.degrees(phi0)

        return phi0
