#!/usr/bin/env python

from dataclasses import dataclass
import numpy as np
from typing import ClassVar
import scipy.constants

from ..beam import Beam
from ..types import NDArray_3, NDArray_3xN, NDArray_N, Parameters
from ..utils import local_to_pointing


@dataclass
class FiniteCylindricalParabolaParams(Parameters):
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
    """

    pointing: NDArray_3xN | NDArray_3
    frequency: NDArray_N | float
    height: NDArray_N | float
    width: NDArray_N | float
    aperture_width: NDArray_N | float

    pointing_shape: ClassVar[tuple[int, ...]] = (3,)
    frequency_shape: ClassVar[None] = None
    height_shape: ClassVar[None] = None
    width_shape: ClassVar[None] = None
    aperture_width_shape: ClassVar[None] = None


class FiniteCylindricalParabola(Beam[FiniteCylindricalParabolaParams]):
    """A finite Cylindrical Parabola with a finite receiver line feed in the
    longitudinal direction, i.e. in the direction of the cylinder axis.

    Custom (measured or more accurately estimated) peak gain at boresight can
    be input, otherwise assumes width (aperture) and height >> wavelength and
    approximates integral with analytic form.

    """

    def __init__(
        self,
        peak_gain: float | None = None,
    ):
        super().__init__()
        self.peak_gain = peak_gain

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

    def copy(self):
        """Return a copy of the current instance."""
        return FiniteCylindricalParabola(
            peak_gain=self.peak_gain,
        )

    def gain(
        self,
        k: NDArray_3xN | NDArray_3,
        parameters: FiniteCylindricalParabolaParams,
    ) -> NDArray_N | float:
        theta, phi = local_to_pointing(k, parameters)
        return self.gain_tf(theta, phi, parameters)

    def gain_tf(
        self,
        theta: NDArray_N | float,
        phi: NDArray_N | float,
        parameters: FiniteCylindricalParabolaParams,
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

        if self.peak_gain is None:
            g0 = self.normalize(
                parameters.aperture_width,
                parameters.height,
                wavelength,
            )
        else:
            g0 = self.peak_gain

        # x = longitudinal angle (i.e. parallel to el.axis), 0 = boresight, radians
        # y = transverse angle, 0 = boresight, radians

        # sinc*sinc is 2D FFT of a rectangular aperture

        # sinc component (longitudinal)
        x = parameters.aperture_width / wavelength * np.sin(phi)

        # sinc component (transverse)
        y = parameters.height / wavelength * np.sin(theta)

        g = np.sinc(x) * np.sinc(y)  # sinc fn. (= field), NB: np.sinc includes pi !!
        g *= np.cos(phi)  # Element gain
        g = g * g  # sinc^2 fn. (= power)

        return g * g0
