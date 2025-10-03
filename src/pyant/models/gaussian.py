#!/usr/bin/env python
import copy

import numpy as np
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
        self.__randn_point = np.array([-0.58617009, 0.29357197, 0.75512921], dtype=np.float64)
        self.peak_gain = peak_gain
        self.min_off_axis = 1e-6

    def copy(self):
        """Return a copy of the current instance."""
        return Gaussian(
            azimuth=copy.deepcopy(self.azimuth),
            elevation=copy.deepcopy(self.elevation),
            frequency=copy.deepcopy(self.frequency),
            I0=copy.deepcopy(self.I0),
            radius=copy.deepcopy(self.parameters["radius"]),
            normal_azimuth=copy.deepcopy(self.normal_azimuth),
            normal_elevation=copy.deepcopy(self.normal_elevation),
            degrees=self.degrees,
        )

    def gain(self, k, ind=None, polarization=None, **kwargs):
        k_len = k.shape[1] if len(k.shape) == 2 else 0
        assert len(k.shape) <= 2, "'k' can only be vectorized with one additional axis"

        params, shape = self.get_parameters(ind, named=True, max_vectors=1)
        assert params["normal"].size == 3, "Cannot vectorize on normal vector"
        assert params["pointing"].size == 3, "Cannot vectorize on pointing"
        if len(params["pointing"].shape) == 2:
            params["pointing"] = params["pointing"].reshape(3)

        params, G = self.broadcast_params(params, shape, k_len)

        lam = scipy.constants.c / params["frequency"]

        pn_dot = np.dot(params["pointing"], params["normal"])
        if np.abs(1 - pn_dot) < self.min_off_axis:
            ct = np.cross(self.__randn_point, params["normal"])
        else:
            ct = np.cross(params["pointing"], params["normal"])

        ct = ct / np.sqrt(np.dot(ct, ct))

        ht = np.cross(params["normal"], ct)
        ht = ht / np.sqrt(np.dot(ht, ht))
        angle = coordinates.vector_angle(params["pointing"], ht, degrees=False)

        ot = np.cross(params["pointing"], ct)
        ot = ot / np.sqrt(np.dot(ot, ot))

        I_1 = np.sin(angle) * self.I0
        a0p = np.sin(angle) * params["radius"]

        sigma1 = 0.7 * a0p / lam
        sigma2 = 0.7 * params["radius"] / lam

        k0 = (k / np.linalg.norm(k, axis=0)).T

        l1 = np.dot(k0, ct).reshape(G.shape)
        m1 = np.dot(k0, ot).reshape(G.shape)

        G = (
            I_1
            * np.exp(-np.pi * l1 * l1 * 2.0 * np.pi * sigma1**2.0)
            * np.exp(-np.pi * m1 * m1 * 2.0 * np.pi * sigma2**2.0)
        )
        return G
