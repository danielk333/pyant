#!/usr/bin/env python

import copy

import numpy as np
import scipy.constants
import scipy.special

from numpy.typing import NDArray
from ..beam import Beam
from .. import coordinates


class Airy(Beam):
    """Airy disk gain model of a radar dish.

    Parameters
    ----------
    pointing : np.ndarray
        Pointing direction of the boresight
    frequency : float | np.ndarray
        Frequency of the radar
    peak_gain : float | np.ndarray
        Peak gain (linear scale) in the pointing direction.
    radius : float | np.ndarray
        Radius in meters of the airy disk

    Notes
    -----
    Singularities
        To avoid singularity at beam center, use
        :math:`\\lim_{x\\mapsto 0} \\frac{J_1(x)}{x} = \\frac{1}{2}` and a threshold.

    """

    def __init__(self, pointing, frequency, radius, peak_gain=1):
        super().__init__()
        self.parameters["pointing"] = pointing
        self.parameters_shape["pointing"] = (3,)
        self.parameters["frequency"] = frequency
        self.parameters["radius"] = radius

        self.peak_gain = peak_gain
        self.zero_limit_eps = 1e-9

        self.validate_parameter_shapes()

    def copy(self):
        """Return a copy of the current instance."""
        beam = Airy(
            pointing=copy.deepcopy(self.parameters["pointing"]),
            frequency=copy.deepcopy(self.parameters["frequency"]),
            radius=copy.deepcopy(self.parameters["radius"]),
            peak_gain=self.peak_gain,
        )
        beam.zero_limit_eps = self.zero_limit_eps
        return beam

    def gain(self, k: NDArray, polarization: NDArray | None = None):
        k_len = self.validate_k_shape(k)
        size = self.size
        scalar_output = size == 0 and k_len == 0

        p = self.parameters["pointing"]
        # size of theta is always k_len or size or a scalar
        theta = coordinates.vector_angle(p, k, degrees=False)

        lam = scipy.constants.c / self.parameters["frequency"]
        k_n = 2.0 * np.pi / lam
        radius = self.parameters["radius"]

        alph = k_n * radius * np.sin(theta)
        if scalar_output:
            alph = np.array([alph])

        inds = alph > self.zero_limit_eps
        not_inds = np.logical_not(inds)

        jn_val = np.empty_like(alph)
        jn_val[inds] = scipy.special.jn(1, alph[inds])

        if scalar_output:
            g = np.empty((1,), dtype=np.float64)
        else:
            g = np.empty((len(alph),), dtype=np.float64)
        g[not_inds] = self.peak_gain
        g[inds] = self.peak_gain * ((2.0 * jn_val[inds] / alph[inds])) ** 2.0

        if scalar_output:
            g = g[0]
        return g
