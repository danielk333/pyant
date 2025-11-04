#!/usr/bin/env python

from dataclasses import dataclass
import numpy as np
from typing import ClassVar
import scipy.constants
import scipy.special

from ..beam import Beam
from ..types import NDArray_3, NDArray_3xN, NDArray_N, Parameters
from ..utils import local_to_pointing


@dataclass
class PhasedFiniteCylindricalParabolaParams(Parameters):
    """
    Parameters
    ----------
    pointing
        Pointing direction of the boresight
    frequency
        Frequency of the radar
    width
        Reflector panel width (axial/azimuth dimension) in meters.
    height
        Reflector panel height (perpendicular/elevation dimension) in meters.
    aperture_width
        Length of the feed in meters. Typical value is same as reflector width.
    depth
        Perpendicular distance from feed to reflector in meters
    phase_steering
        Phase steering (NOTE: radians) angle applied to the feed bridge of antennas
    """

    pointing: NDArray_3xN | NDArray_3
    frequency: NDArray_N | float
    height: NDArray_N | float
    width: NDArray_N | float
    aperture_width: NDArray_N | float
    phase_steering: NDArray_N | float
    depth: NDArray_N | float

    pointing_shape: ClassVar[tuple[int, ...]] = (3,)
    frequency_shape: ClassVar[None] = None
    height_shape: ClassVar[None] = None
    width_shape: ClassVar[None] = None
    aperture_width_shape: ClassVar[None] = None
    phase_steering_shape: ClassVar[None] = None
    depth_shape: ClassVar[None] = None


class PhasedFiniteCylindricalParabola(Beam[PhasedFiniteCylindricalParabolaParams]):
    """A finite Cylindrical Parabola with a phased finite receiver line feed
        in the longitudinal direction, i.e. in the direction of the cylinder axis.

    Notes
    ------
    Peak gain
        Custom (measured or more accurately estimated) peak gain at boresight can
        be input, otherwise assumes width and height >> wavelength and approximates
        integral with analytic form.

    """

    def __init__(
        self,
        peak_gain: float | None = None,
    ):
        super().__init__()
        self.peak_gain = peak_gain

    def copy(self):
        """Return a copy of the current instance."""
        return PhasedFiniteCylindricalParabola(
            peak_gain=self.peak_gain,
        )

    def normalize(
        self,
        width: NDArray_N | float,
        height: NDArray_N | float,
        wavelength: NDArray_N | float,
    ) -> NDArray_N | float:
        """Calculate normalization constant for beam pattern by assuming
        width and height >> wavelength.
        """
        return 4 * np.pi * width * height / wavelength**2

    def gain(
        self,
        k: NDArray_3xN | NDArray_3,
        parameters: PhasedFiniteCylindricalParabolaParams,
    ) -> NDArray_N | float:
        theta, phi = local_to_pointing(k, parameters)
        return self.gain_tf(theta, phi, parameters)

    def gain_tf(
        self,
        theta: NDArray_N | float,
        phi: NDArray_N | float,
        parameters: PhasedFiniteCylindricalParabolaParams,
    ) -> NDArray_N | float:
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
        wavelength = scipy.constants.c / parameters.frequency
        phase_steering = parameters.phase_steering

        # Compute effective area loss due to spillover when phase-steering past edge of reflector.
        A = (
            parameters.depth * np.tan(np.abs(phase_steering))
            - (parameters.width - parameters.aperture_width) / 2
        )
        w_loss = np.clip(A, 0, None)

        # This implies geometric optics for the feed-to-reflector path
        w_eff = np.clip(parameters.aperture_width - w_loss, 0, None)

        if self.peak_gain is None:
            g0 = self.normalize(w_eff, parameters.height, wavelength)
        else:
            g0 = self.peak_gain

        # y = longitudinal angle (i.e. parallel to el.axis), 0 = boresight, radians
        # x = transverse angle, 0 = boresight, radians

        # sinc component (transverse)
        x = w_eff / wavelength * (np.sin(phi) - phase_steering)

        # sinc component (longitudinal)
        y = parameters.height / wavelength * np.sin(theta)

        # sinc fn. (= field), NB: np.sinc includes pi !!
        g = np.sinc(x) * np.sinc(y) * (w_eff / parameters.aperture_width)

        # NB! This factor of cos(phi) is NOT geometric foreshortening --
        # It is instead a crude model for element (i.e., dipole) gain!
        g *= np.cos(phi)

        g = g * g  # sinc^2 fn. (= power)

        return g * g0

    def _nominal_phase_steering(self, phi, degrees=False):
        """
        For a given desired pointing `phi`, compute the nominal phase
        steering angle that gives a pattern that peaks at `phi`.

        Always returns radians

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
        return phi0
