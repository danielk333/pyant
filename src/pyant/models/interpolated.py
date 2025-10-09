import copy
from typing import Literal
import numpy as np
from numpy.typing import NDArray
import scipy.interpolate

from ..beam import Beam
from ..coordinates import compute_k_grid

InterpMethods = Literal["bivariate_spline", "linear"]

class Interpolated(Beam):
    """Interpolated gain pattern. Does not assume any effect on pointing.
    """

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
        beam,
        interpolation_method: InterpMethods = "linear",
        polarization=None,
        centered=True,
        min_elevation=0.0,
        resolution=1000,
    ):
        if beam.size > 0:
            raise ValueError(
                "Can only interpolate beam with scalar parameters -"
                f"dont know which of the {beam.size} options to pick"
            )
        if "pointing" not in beam.parameters:
            pointing = np.array([0, 0, 1], dtype=np.float64)
        else:
            pointing = beam.parameters["pointing"]

        cmin = np.cos(np.radians(min_elevation))
        S, K, k, inds, kx, ky = compute_k_grid(pointing, resolution, centered, cmin)
        S = np.zeros_like(S)
        S[inds] = beam.gain(k[:, inds], polarization=polarization)

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

    def gain(self, k: NDArray, polarization: NDArray | None = None):
        k_ = k / np.linalg.norm(k, axis=0)
        g = self.interpolated(k_[0, ...], k_[1, ...], **self._call_kw)
        return g
