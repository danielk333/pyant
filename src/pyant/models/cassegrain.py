#!/usr/bin/env python
import copy

import numpy as np
import scipy.special

from ..beam import Beam
from .. import coordinates


def calculate_cassegrain_AB(theta, lam, a0, a1):
    """Calculates the A and B parameters in the Cassegrain gain model"""
    scale = lam / (np.pi * np.sin(theta))
    A = (scale / (a0**2 - a1**2)) ** 2
    ba0 = a0 * scipy.special.jn(1, a0 / scale)
    ba1 = a1 * scipy.special.jn(1, a1 / scale)
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

    def __init__(self, azimuth, elevation, frequency, I0, outer_radius, inner_radius, **kwargs):
        super().__init__(azimuth, elevation, frequency, **kwargs)
        self.I0 = I0
        self.eps = 1e-6
        self.min_off_axis = 1e-9

        self.register_parameter("outer_radius")
        self.register_parameter("inner_radius")

        self.fill_parameter("outer_radius", outer_radius)
        self.fill_parameter("inner_radius", inner_radius)

    def copy(self):
        """Return a copy of the current instance."""
        return Cassegrain(
            azimuth=copy.deepcopy(self.azimuth),
            elevation=copy.deepcopy(self.elevation),
            frequency=copy.deepcopy(self.frequency),
            I0=copy.deepcopy(self.I0),
            a1=copy.deepcopy(self.parameters["a1"]),
            a0=copy.deepcopy(self.parameters["a0"]),
            degrees=self.degrees,
        )

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
                theta = np.broadcast_to(theta, G.shape)

        lam = scipy.constants.c / params["frequency"]

        # pointings not close to zero off-axis angle
        inds = np.pi * np.sin(theta) > self.min_off_axis

        if len(G.shape) == 0:
            if not inds:
                G = self.I0
            else:
                a0, a1 = params["outer_radius"], params["inner_radius"]

                A, B = calculate_cassegrain_AB(theta, lam, a0, a1)
                A_eps, B_eps = calculate_cassegrain_AB(self.eps, lam, a0, a1)

                G = self.I0 * (A * B) / (A_eps * B_eps)
        else:
            a0, a1 = params["outer_radius"][inds], params["inner_radius"][inds]

            A, B = calculate_cassegrain_AB(theta[inds], lam[inds], a0, a1)
            A_eps, B_eps = calculate_cassegrain_AB(self.eps, lam[inds], a0, a1)

            G[np.logical_not(inds)] = self.I0
            G[inds] = self.I0 * (A * B) / (A_eps * B_eps)

        return G
