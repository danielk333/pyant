#!/usr/bin/env python

import copy

import numpy as np
import scipy.constants
import scipy.special

from ..beam import Beam
from .. import coordinates


class Airy(Beam):
    """Airy disk gain model of a radar dish.

    Parameters
    ----------
    I0 : float
        Peak gain (linear scale) in the pointing direction.
    radius : float
        Radius in meters of the airy disk

    Notes
    -----
    Singularities
        To avoid singularity at beam center, use
        :math:`\\lim_{x\\mapsto 0} \\frac{J_1(x)}{x} = \\frac{1}{2}` and a threshold.

    """

    def __init__(self, azimuth, elevation, frequency, I0, radius, **kwargs):
        super().__init__(azimuth, elevation, frequency, **kwargs)
        self.I0 = I0
        self.register_parameter("radius")
        self.fill_parameter("radius", radius)

    def copy(self):
        """Return a copy of the current instance."""
        return Airy(
            azimuth=copy.deepcopy(self.azimuth),
            elevation=copy.deepcopy(self.elevation),
            frequency=copy.deepcopy(self.frequency),
            I0=copy.deepcopy(self.I0),
            radius=copy.deepcopy(self.parameters["radius"]),
            degrees=self.degrees,
        )

    @property
    def radius(self):
        """Radius in meters of the airy disk"""
        return self.parameters["radius"]

    @radius.setter
    def radius(self, val):
        self.fill_parameter("radius", val)

    def gain(self, k, ind=None, polarization=None, **kwargs):
        k_len = k.shape[1] if len(k.shape) == 2 else 0
        assert len(k.shape) <= 2, "'k' can only be vectorized with one additional axis"

        params, shape = self.get_parameters(ind, named=True, max_vectors=1)
        params, G = self.broadcast_params(params, shape, k_len)

        p_len = params["pointing"].shape[1] if len(params["pointing"].shape) == 2 else 0
        if p_len > 1 and k_len > 1:
            theta = np.empty_like(G)
            for ind in range(p_len):
                theta[:, ind] = coordinates.vector_angle(
                    params["pointing"][:, ind],
                    k,
                    degrees=False,
                )
        else:
            if p_len == 1:
                params["pointing"] = params["pointing"].reshape(3)
            theta = coordinates.vector_angle(params["pointing"], k, degrees=False)
            if theta.size == G.size:
                if len(theta.shape) > 0:
                    theta.shape = G.shape
            else:
                theta = theta.reshape(theta.size, 1)
                theta = np.broadcast_to(theta, G.shape)

        lam = scipy.constants.c / params["frequency"]
        k_n = 2.0 * np.pi / lam
        alph = k_n * params["radius"] * np.sin(theta)
        jn_val = scipy.special.jn(1, alph)
        inds = alph < 1e-9

        if len(G.shape) == 0:
            if inds:
                G = self.I0
            else:
                G = self.I0 * ((2.0 * jn_val / alph)) ** 2.0
        else:
            not_inds = np.logical_not(inds)
            G[inds] = self.I0
            G[not_inds] = self.I0 * ((2.0 * jn_val[not_inds] / alph[not_inds])) ** 2.0

        return G
