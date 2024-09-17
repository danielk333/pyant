import copy

import numpy as np
import scipy.interpolate

from ..beam import Beam
from ..plotting import compute_k_grid


class Interpolated(Beam):
    """Interpolated gain pattern. Does not assume any effect on pointing.

    Parameters
    ----------
    scaling : float
        Scaling of the gain pattern to apply.

    Attributes
    ----------
    scaling : float
        Scaling of the gain pattern to apply.

    """

    def __init__(self, azimuth, elevation, frequency, scaling=1.0, **kwargs):
        super().__init__(azimuth, elevation, frequency, **kwargs)
        self.interpolated = None
        self.scaling = scaling

    def copy(self):
        """Return a copy of the current instance."""
        bm = Interpolated(
            azimuth=copy.deepcopy(self.azimuth),
            elevation=copy.deepcopy(self.elevation),
            frequency=copy.deepcopy(self.frequency),
            scaling=copy.deepcopy(self.scaling),
            degrees=self.degrees,
        )
        bm.interpolated = self.interpolated
        return bm

    def generate_interpolation(
        self,
        beam,
        ind=None,
        polarization=None,
        centered=True,
        min_elevation=0.0,
        resolution=1000,
    ):
        params, shape = beam.get_parameters(ind, named=True)
        pointing = params["pointing"]

        cmin = np.cos(np.radians(min_elevation))
        S, K, k, inds, kx, ky = compute_k_grid(pointing, resolution, centered, cmin)
        S = np.zeros_like(S)
        S[inds] = beam.gain(k[:, inds], polarization=polarization, ind=ind).flatten()
        S = S.reshape(resolution, resolution)

        self.interpolated = scipy.interpolate.RectBivariateSpline(kx, ky, S.T)

    def save(self, fname):
        np.save(fname, self.interpolated)

    def load(self, fname):
        f_obj = np.load(fname, allow_pickle=True)
        self.interpolated = f_obj.item()

    def gain(self, k, ind=None, polarization=None, **kwargs):
        k_ = k / np.linalg.norm(k, axis=0)

        G = self.interpolated(k_[0, ...], k_[1, ...], grid=False)

        if len(k.shape) == 1:
            if len(G.shape) > 0:
                G = G[0]
        return G * self.scaling
