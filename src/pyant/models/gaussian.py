#!/usr/bin/env python

from dataclasses import dataclass
from typing import ClassVar
import numpy as np
from numpy.typing import NDArray
import scipy.constants
import scipy.special
import spacecoords.linalg as linalg

from ..beam import Beam, get_and_validate_k_shape
from ..types import NDArray_3, NDArray_3xN, NDArray_N, Parameters


@dataclass
class GaussianParams(Parameters):
    """
    Parameters
    ----------
    pointing
        Pointing direction of the array by phasing
    normal_pointing
        Pointing direction of the planar array
    frequency
        Frequency of the radar
    radius
        Radius of array
    filling_factor
        Amount of space on average that collects radiation inside the aperture
    """

    pointing: NDArray_3xN | NDArray_3
    normal_pointing: NDArray_3xN | NDArray_3
    frequency: NDArray_N | float
    radius: NDArray_N | float
    filling_factor: NDArray_N | float

    pointing_shape: ClassVar[tuple[int, ...]] = (3,)
    normal_pointing_shape: ClassVar[tuple[int, ...]] = (3,)
    frequency_shape: ClassVar[None] = None
    radius_shape: ClassVar[None] = None
    filling_factor_shape: ClassVar[None] = None


class Gaussian(Beam[GaussianParams]):
    """Gaussian tapered planar array model

    Parameters
    ----------
    peak_gain
        Peak gain (linear scale) in the pointing direction.
    min_off_axis
        Minimum off axis angle where we instead use the limit value
    """

    def __init__(
        self,
        peak_gain: float = 1,
        min_off_axis: float = 1e-9,
    ):
        super().__init__()
        # Random number in case pointing and planar normal align
        # Used to determine basis vectors in the plane perpendicular to pointing
        self._randn_point = np.array([-0.58617009, 0.29357197, 0.75512921], dtype=np.float64)
        self.min_off_axis = min_off_axis
        self.peak_gain = peak_gain

    def copy(self):
        """Return a copy of the current instance."""
        beam = Gaussian(
            peak_gain=self.peak_gain,
            min_off_axis=self.min_off_axis,
        )
        beam._randn_point = self._randn_point.copy()

        return beam

    def gain(self, k: NDArray_3xN | NDArray_3, parameters: GaussianParams) -> NDArray_N | float:
        size = parameters.size()
        k_len = get_and_validate_k_shape(size, k)
        if k_len == 0:
            return np.empty((0,), dtype=k.dtype)
        scalar_output = size == 0 and k_len is None
        # >>> import scipy.constants as c
        # >>> c.c / 3.4e-9
        # 8.817425235294118e+16
        # >>> 3.4e-9 / c.c
        # 1.1341179236737169e-17
        # >>> 3.4e9 / c.c
        # 11.341179236737169
        # >>> c.c / 3.4e9 
        # 0.08817425235294117
        # >>> (c.c / 3.4e9)*1e-3
        # 8.817425235294117e-05
        # >>> (c.c / 3.4e9)*1e3
        # 88.17425235294117
        # >>> (c.c / 3.4e9)*1e2
        # 8.817425235294117
        # >>> (c.c / 1.2e9)*1e2
        # 24.982704833333333
        # >>> (c.c / 3.4e9)*1e2
        # 8.817425235294117
        # >>> lam = c.c / 3.4e9
        # >>> alpha = 2
        # >>> s = alpha * lam / 2
        # >>> s
        # 0.08817425235294117
        # >>> N = int(500e3 / 500)
        # >>> N
        # 1000
        # >>> N = int(500e3 / 50)
        # >>> N
        # 10000
        # >>> import numpy as np
        # >>> d = np.sqrt(4*N/np.pi)*s
        # >>> d
        # np.float64(9.94939894292813)
        # >>> theta = lam / d
        # >>> np.degrees(theta)
        # np.float64(0.5077706251929806)
        pointing = self.parameters["pointing"]
        normal = self.parameters["normal_pointing"]
        lam = scipy.constants.c / self.parameters["frequency"]
        radius = self.parameters["radius"]

        pn_dot = np.sum(pointing * normal, axis=0)
        inds = np.abs(1 - pn_dot) < self.min_off_axis
        if size == 0:
            if inds:
                ct = np.cross(self._randn_point, normal)
            else:
                ct = np.cross(pointing, normal)
        else:
            not_inds = np.logical_not(inds)
            ct = np.empty_like(normal)
            ct[:, inds] = np.linalg.cross(self._randn_point, normal[:, inds], axis=0)
            ct[:, not_inds] = np.linalg.cross(pointing[:, not_inds], normal[:, not_inds], axis=0)

        ct = ct / np.linalg.norm(ct, axis=0)

        ht = np.linalg.cross(normal, ct, axis=0)
        ht = ht / np.linalg.norm(ht, axis=0)
        angle = coordinates.vector_angle(pointing, ht, degrees=False)

        ot = np.cross(pointing, ct, axis=0)
        ot = ot / np.linalg.norm(ot, axis=0)

        peak_1 = np.sin(angle) * self.peak_gain
        a0p = np.sin(angle) * radius

        sigma1 = 0.7 * a0p / lam
        sigma2 = 0.7 * radius / lam

        k0 = k / np.linalg.norm(k, axis=0)
        if size > 0 and k_len == 0:
            k0 = np.broadcast_to(k0.reshape(3, 1), (3, size))
        elif k_len > 0 and size == 0:
            ct = np.broadcast_to(ct.reshape(3, 1), (3, k_len))
            ot = np.broadcast_to(ot.reshape(3, 1), (3, k_len))

        l1 = np.sum(k0 * ct, axis=0)
        m1 = np.sum(k0 * ot, axis=0)

        g = (
            peak_1
            * np.exp(-np.pi * l1 * l1 * 2.0 * np.pi * sigma1**2.0)
            * np.exp(-np.pi * m1 * m1 * 2.0 * np.pi * sigma2**2.0)
        )
        return g
