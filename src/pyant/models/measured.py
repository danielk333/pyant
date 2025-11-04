#!/usr/bin/env python

import copy
from dataclasses import dataclass
from typing import ClassVar, Literal

import scipy.interpolate as sci
import numpy as np
import spacecoords.linalg as linalg

from ..beam import Beam, get_and_validate_k_shape
from ..types import NDArray_3, NDArray_3xN, NDArray_N, Parameters

InterpMethods = Literal["cubic_spline", "linear"]


@dataclass
class MeasuredAzimuthallySymmetricParams(Parameters):
    """
    Parameters
    ----------
    pointing
        Pointing direction of the boresight
    """

    pointing: NDArray_3xN | NDArray_3

    pointing_shape: ClassVar[tuple[int, ...]] = (3,)


class MeasuredAzimuthallySymmetric(Beam[MeasuredAzimuthallySymmetricParams]):
    """An interpolation of a measured 1d gain pattern"""

    def __init__(
        self,
        off_axis_angle: NDArray_N,
        gains: NDArray_N,
        interpolation_method: InterpMethods = "linear",
        degrees: bool = True,
    ):
        super().__init__()
        self.off_axis_angle = off_axis_angle
        self.gains = gains
        self.interpolation_method = interpolation_method
        self.degrees = degrees

    def gain(
        self,
        k: NDArray_3xN | NDArray_3,
        parameters: MeasuredAzimuthallySymmetricParams,
    ) -> NDArray_N | float:
        size = parameters.size()
        k_len = get_and_validate_k_shape(size, k)
        if k_len == 0:
            return np.empty((0,), dtype=k.dtype)

        phi = linalg.vector_angle(k, parameters.pointing, degrees=self.degrees)

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
            off_axis_angle=copy.deepcopy(self.off_axis_angle),
            gains=copy.deepcopy(self.gains),
            interpolation_method=self.interpolation_method,
            degrees=self.degrees,
        )
