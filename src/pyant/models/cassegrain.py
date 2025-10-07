#!/usr/bin/env python
import copy

import numpy as np
from numpy.typing import NDArray
import scipy.special

from ..beam import Beam
from .. import coordinates


def calculate_cassegrain_AB(theta, lam, a0, a1):
    """Calculates the A and B parameters in the Cassegrain gain model"""
    scale = lam / (np.pi * np.sin(theta))
    A = (scale / (a0**2 - a1**2)) ** 2
    ba0 = a0 * scipy.special.j1(a0 / scale)
    ba1 = a1 * scipy.special.j1(a1 / scale)
    B = (ba0 - ba1) ** 2

    return A, B


class Cassegrain(Beam):
    """Cassegrain gain model of a radar dish.

    Parameters
    ----------
    I0 : float
        Peak gain (linear scale) in the pointing direction.
    outer_radius : float
        Radius of main reflector
    inner_radius : float
        Radius of sub reflector

    Notes
    -----
    Derivation
        The gain pattern is expressed as the Fourier transform of an annular
        region with outer radius a0 and inner radius a1.  Values below the
        aperture plane or below the horizon should not be believed.
    """

    def __init__(self, pointing, frequency, outer_radius, inner_radius, peak_gain=1):
        super().__init__()
        self.parameters["pointing"] = pointing
        self.parameters_shape["pointing"] = (3,)
        self.parameters["frequency"] = frequency
        self.parameters["outer_radius"] = outer_radius
        self.parameters["inner_radius"] = inner_radius
        self.eps = 1e-6
        self.min_off_axis = 1e-9
        self.peak_gain = peak_gain
        self.validate_parameter_shapes()

    def copy(self):
        """Return a copy of the current instance."""
        beam = Cassegrain(
            pointing=copy.deepcopy(self.parameters["pointing"]),
            frequency=copy.deepcopy(self.parameters["frequency"]),
            outer_radius=copy.deepcopy(self.parameters["outer_radius"]),
            inner_radius=copy.deepcopy(self.parameters["inner_radius"]),
            peak_gain=self.peak_gain,
        )
        beam.eps = self.eps
        beam.min_off_axis = self.min_off_axis

        return beam

    def gain(self, k: NDArray, polarization: NDArray | None = None):
        k_len = self.validate_k_shape(k)
        size = self.size
        scalar_output = size == 0 and k_len == 0

        p = self.parameters["pointing"]
        theta = coordinates.vector_angle(p, k, degrees=False)

        lam = scipy.constants.c / self.parameters["frequency"]
        a0 = self.parameters["outer_radius"]
        a1 = self.parameters["inner_radius"]

        if scalar_output:
            theta = np.array([theta])
        # pointings not close to zero off-axis angle
        inds = np.pi * np.sin(theta) > self.min_off_axis

        if size > 0:
            a0, a1, lam = a0[inds], a1[inds], lam[inds]

        if scalar_output:
            g = np.empty((1,), dtype=np.float64)
        else:
            g = np.empty((len(theta),), dtype=np.float64)

        A, B = calculate_cassegrain_AB(theta[inds], lam, a0, a1)
        A_eps, B_eps = calculate_cassegrain_AB(self.eps, lam, a0, a1)

        g[np.logical_not(inds)] = self.peak_gain
        g[inds] = self.peak_gain * (A * B) / (A_eps * B_eps)

        if scalar_output:
            g = g[0]
        return g
