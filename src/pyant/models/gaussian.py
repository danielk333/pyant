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
    beam_width_scaling: NDArray_N | float

    pointing_shape: ClassVar[tuple[int, ...]] = (3,)
    normal_pointing_shape: ClassVar[tuple[int, ...]] = (3,)
    frequency_shape: ClassVar[None] = None
    radius_shape: ClassVar[None] = None
    beam_width_scaling_shape: ClassVar[None] = None


class Gaussian(Beam[GaussianParams]):
    """Gaussian tapered planar array model. The model uses two Gaussian functions, one along the
    latitude line in the azimuthal direction of pointing which is dependant on the angle

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
        pointing = parameters.pointing
        normal = parameters.normal_pointing
        lam = scipy.constants.c / parameters.frequency
        radius = parameters.radius
        alph = parameters.beam_width_scaling

        pn_dot = np.sum(pointing * normal, axis=0)
        inds = np.abs(1 - pn_dot) < np.sin(self.min_off_axis)
        if size is None:
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

        ot = np.cross(pointing, ct, axis=0)
        ot = ot / np.linalg.norm(ot, axis=0)

        antenna_element_scaling = pn_dot * self.peak_gain

        # solve exp(-2 pi^2 sin^2(lam/d) sig^2)) = 0.5
        # comes from the definition of half-power at theta = lambda / d
        sigma_lat = np.sqrt(
            np.log(2) / (2 * pn_dot * np.pi**2 * np.sin(alph * lam / (radius * 4)) ** 2)
        )
        sigma_lon = np.sqrt(np.log(2) / (2 * np.pi**2 * np.sin(alph * lam / (radius * 4)) ** 2))

        k0 = k / np.linalg.norm(k, axis=0)
        if size is not None and k_len is None:
            k0 = np.broadcast_to(k0.reshape(3, 1), (3, size))
        elif k_len is not None and size is None:
            ct = np.broadcast_to(ct.reshape(3, 1), (3, k_len))
            ot = np.broadcast_to(ot.reshape(3, 1), (3, k_len))

        l1 = np.sum(k0 * ct, axis=0)
        m1 = np.sum(k0 * ot, axis=0)

        g = antenna_element_scaling * np.exp(
            -2 * np.pi**2 * ((l1 * sigma_lat) ** 2 + (m1 * sigma_lon) ** 2)
        )
        return g
