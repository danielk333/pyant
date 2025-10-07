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
        pointing,
        frequency,
        width,
        height,
        aperture_width,
        phase_steering,
        depth,
        peak_gain=None,
        degrees=False,
    ):
        super().__init__(
            pointing=pointing,
            frequency=frequency,
            width=width,
            height=height,
            aperture_width=aperture_width,
            peak_gain=peak_gain,
        )
        self.parameters["phase_steering"] = phase_steering
        self.parameters["depth"] = depth
        self.degrees = degrees
        self.validate_parameter_shapes()

    def copy(self):
        """Return a copy of the current instance."""
        return PhasedFiniteCylindricalParabola(
            pointing=copy.deepcopy(self.parameters["pointing"]),
            phase_steering=copy.deepcopy(self.parameters["phase_steering"]),
            frequency=copy.deepcopy(self.parameters["frequency"]),
            height=copy.deepcopy(self.parameters["height"]),
            width=copy.deepcopy(self.parameters["width"]),
            depth=copy.deepcopy(self.parameters["depth"]),
            aperture_width=copy.deepcopy(self.parameters["aperture_width"]),
            peak_gain=self.peak_gain,
            degrees=self.degrees,
        )

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

        if self.degrees:
            phase_steering = np.radians(self.parameters["phase_steering"])
        else:
            phase_steering = self.parameters["phase_steering"]

        # Compute effective area loss due to spillover when phase-steering past edge of reflector.
        A = (
            self.parameters["depth"] * np.tan(np.abs(phase_steering))
            - (self.parameters["width"] - self.parameters["aperture_width"]) / 2
        )
        w_loss = np.clip(A, 0, None)

        # This implies geometric optics for the feed-to-reflector path
        w_eff = np.clip(self.parameters["aperture_width"] - w_loss, 0, None)

        if self.peak_gain is None:
            g0 = self.normalize(w_eff, self.parameters["height"], wavelength)
        else:
            g0 = self.peak_gain

        # y = longitudinal angle (i.e. parallel to el.axis), 0 = boresight, radians
        # x = transverse angle, 0 = boresight, radians

        # sinc component (transverse)
        x = w_eff / wavelength * (np.sin(phi) - phase_steering)

        # sinc component (longitudinal)
        y = self.parameters["height"] / wavelength * np.sin(theta)

        # sinc fn. (= field), NB: np.sinc includes pi !!
        g = np.sinc(x) * np.sinc(y) * (w_eff / self.parameters["aperture_width"])

        # NB! This factor of cos(phi) is NOT geometric foreshortening --
        # It is instead a crude model for element (i.e., dipole) gain!
        g *= np.cos(phi)

        g = g * g  # sinc^2 fn. (= power)

        return g * g0

    def _nominal_phase_steering(self, phi, degrees=False):
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
        if degrees:
            phi = np.radians(phi)
        assert np.all(np.abs(phi) <= np.pi / 2), f"cannot steer past the horizon {np.abs(phi)}"

        phi0 = np.sin(phi)

        if degrees:
            phi0 = np.degrees(phi0)

        return phi0
