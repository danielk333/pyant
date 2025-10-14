#!/usr/bin/env python

import copy
from typing import Literal
from ..types import NDArray_3, NDArray_3xN, NDArray_N

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
        pointing: NDArray_3 | NDArray_3xN,
        off_axis_angle: NDArray_N,
        gains: NDArray_N,
        interpolation_method: InterpMethods = "linear",
        degrees: bool = True,
    ):
        super().__init__()
        self.parameters["pointing"] = pointing
        self.parameters_shape["pointing"] = (3,)

        self.off_axis_angle = off_axis_angle
        self.gains = gains
        self.interpolation_method = interpolation_method
        self.degrees = degrees

    def gain(self, k: NDArray, polarization: NDArray | None = None):
        phi = coordinates.vector_angle(k, self.parameters["pointing"], degrees=self.degrees)

        if self.interpolation_method == "cubic_spline":
            cbs = sci.CubicSpline(self.off_axis_angle, self.gains, extrapolate=False)
            g = cbs(phi)
        elif self.interpolation_method == "linear":
            g = np.interp(phi, self.off_axis_angle, self.gains, left=np.nan, right=np.nan)
        else:
            raise ValueError(
                f"Interpolation method '{self.interpolation_method}' not supported,"
                f"{InterpMethods} are avalible",
            )
        return g

    def copy(self):
        return MeasuredAzimuthallySymmetric(
            pointing=copy.deepcopy(self.parameters["pointing"]),
            off_axis_angle=copy.deepcopy(self.off_axis_angle),
            gains=copy.deepcopy(self.gains),
            interpolation_method=self.interpolation_method,
            degrees=self.degrees,
        )
