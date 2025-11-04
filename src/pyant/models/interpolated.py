#!/usr/bin/env python

from dataclasses import dataclass
import copy
from typing import Literal
import numpy as np
import scipy.interpolate

from ..beam import Beam
from ..types import NDArray_3, NDArray_3xN, NDArray_N, Parameters
from ..utils import compute_k_grid

InterpMethods = Literal["bivariate_spline", "linear"]


@dataclass
class InterpolatedParams(Parameters):
    """placeholder"""

    pass


class Interpolated(Beam[InterpolatedParams]):
    """Interpolated gain pattern. Does not assume any effect on pointing."""

    def __init__(self):
        super().__init__()
        self.interpolated = None
        self.interpolation_method = None
        self._call_kw = None

    def copy(self):
        """Return a copy of the current instance."""
        bm = Interpolated()
        bm.interpolated = copy.deepcopy(self.interpolated)
        bm.interpolation_method = self.interpolation_method
        bm._call_kw = copy.deepcopy(self.interpolated)
        return bm

    def generate_interpolation(
        self,
        beam: Beam,
        parameters: Parameters,
        interpolation_method: InterpMethods = "linear",
        centered: bool = True,
        min_elevation: float = 0.0,
        resolution: int = 1000,
    ):
        if not isinstance(beam, Beam):
            raise TypeError(f"Can only interpolate Beam, not '{type(beam)}'")

        if parameters.size is None:
            raise ValueError(
                "Can only plot beam with scalar parameters -"
                f"dont know which of the {parameters.size} options to pick"
            )

        if "pointing" not in parameters.keys:
            pointing = np.array([0, 0, 1], dtype=np.float64)
        else:
            pointing = parameters.pointing  # type: ignore

        cmin = np.cos(np.radians(min_elevation))
        S, K, k, inds, kx, ky = compute_k_grid(pointing, resolution, centered, cmin)
        S = np.zeros_like(S)
        S[inds] = beam.gain(k[:, inds], parameters)

        self._call_kw = {}
        if interpolation_method == "linear":
            self.interpolated = scipy.interpolate.LinearNDInterpolator(k[:2, :].T, S)
        elif interpolation_method == "bivariate_spline":
            S = S.reshape(resolution, resolution)
            self._call_kw["grid"] = False
            self.interpolated = scipy.interpolate.RectBivariateSpline(kx, ky, S.T)

        else:
            raise ValueError(
                f"Interpolation method '{interpolation_method}' not supported,"
                f"{InterpMethods} are avalible",
            )
        self.interpolation_method = interpolation_method

    def save(self, fname):
        np.save(fname, self.interpolated)

    def load(self, fname):
        f_obj = np.load(fname, allow_pickle=True)
        self.interpolated = f_obj.item()

    def gain(self, k: NDArray_3xN | NDArray_3, parameters: InterpolatedParams) -> NDArray_N | float:
        g = self.interpolated(k[0, ...], k[1, ...], **self._call_kw)
        return g
