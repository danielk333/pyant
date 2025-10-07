#!/usr/bin/env python
import copy

import numpy as np
from numpy.typing import NDArray
import scipy.constants
import scipy.special

from ..beam import Beam
from .. import coordinates


class Gaussian(Beam):
    """Gaussian tapered planar array model
    TODO: docstring

    Parameters
    ----------
    I0 : float
        Peak gain (linear scale) in the pointing direction.
    radius : float
        Radius in meters of the planar array
    normal_azimuth : float
        Azimuth of normal vector of the planar array in degrees.
    normal_elevation : float
        Elevation of pointing direction in degrees.

    Attributes
    ----------
    I0 : float
        Peak gain (linear scale) in the pointing direction.
    radius : float
        Radius in meters of the airy disk
    normal : numpy.ndarray
        Planar array normal vector in local coordinates
    normal_azimuth : float
        Azimuth of normal vector of the planar array in degrees.
    normal_elevation : float
        Elevation of pointing direction in degrees.
    """

    def __init__(self, pointing, frequency, radius, normal_pointing, peak_gain=1):
        super().__init__()
        self.parameters["pointing"] = pointing
        self.parameters_shape["pointing"] = (3,)
        self.parameters["frequency"] = frequency
        self.parameters["radius"] = radius
        self.parameters["normal_pointing"] = normal_pointing
        self.parameters_shape["normal_pointing"] = (3,)

        # Random number in case pointing and planar normal align
        # Used to determine basis vectors in the plane perpendicular to pointing
        self._randn_point = np.array([-0.58617009, 0.29357197, 0.75512921], dtype=np.float64)
        self.peak_gain = peak_gain
        self.min_off_axis = 1e-6
        self.validate_parameter_shapes()

    def copy(self):
        """Return a copy of the current instance."""
        beam = Gaussian(
            pointing=copy.deepcopy(self.parameters["pointing"]),
            frequency=copy.deepcopy(self.parameters["frequency"]),
            radius=copy.deepcopy(self.parameters["radius"]),
            normal_pointing=copy.deepcopy(self.parameters["normal_pointing"]),
            peak_gain=self.peak_gain,
        )
        beam._randn_point = self._randn_point.copy()
        beam.min_off_axis = self.min_off_axis

        return beam

    def gain(self, k: NDArray, polarization: NDArray | None = None):
        k_len = self.validate_k_shape(k)
        size = self.size

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
