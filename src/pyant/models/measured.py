#!/usr/bin/env python

import copy
from typing import Literal

import scipy.interpolate as sci
import numpy as np
from numpy.typing import NDArray

from .. import coordinates
from ..beam import Beam

InterpMethods = Literal["cubic_spline", "linear"]


class MeasuredAzimuthallySymmetric(Beam):
    """An interpolation of a measured 1d gain pattern"""

    def __init__(
        self,
        elevations: NDArray,
        gains: NDArray,
        interpolation_method: InterpMethods = "linear",
        degrees: bool = True,
    ):
        super().__init__()
        self.elevations = elevations
        self.gains = gains
        self.interpolation_method = interpolation_method
        self.degrees = degrees

    def gain(self, k: NDArray, polarization: NDArray | None = None):
        azelr = coordinates.cart_to_sph(k, degrees=self.degrees)
        els = azelr[1, ...]

        if self.interpolation_method == "cubic_spline":
            cbs = sci.CubicSpline(self.elevations, self.gains, extrapolate=False)
            g = cbs(els)
        elif self.interpolation_method == "linear":
            g = np.interp(els, self.elevations, self.gains, left=np.nan, right=np.nan)
        else:
            raise ValueError(
                f"Interpolation method '{self.interpolation_method}' not supported,"
                f"{InterpMethods} are avalible",
            )
        return g

    def copy(self):
        return MeasuredAzimuthallySymmetric(
            elevations=copy.deepcopy(self.elevations),
            gains=copy.deepcopy(self.gains),
            interpolation_method=self.interpolation_method,
            degrees=self.degrees,
        )
