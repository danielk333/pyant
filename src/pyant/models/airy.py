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
            radius=copy.deepcopy(self.radius),
            degrees=self.degrees,
        )

    def gain(self, k, ind=None, polarization=None, **kwargs):
        k_len = k.shape[1] if len(k.shape) == 2 else 0
        assert len(k.shape) <= 2, "'k' can only be vectorized with one additional axis"

        params, G = self.get_parameters(ind, named=True, generate_array=True, extend_size=k_len)
        bcast = self._get_broadcaster(named=True, extend_size=k_len)
        # CHANGE ALL OF THIS: dont try to be too fancy, thats the problem
        # 1. get size of parameters
        # 2. repmat the parameters to the size of G
        # 3. then just do calculations without broadcasting
        # (ram should not be a problem here and its fast as well)
        params = {key: np.array([x]) if len(x.shape) == 0 else x for key, x in params.items()}
        if len(params["pointing"].shape) == 1:
            params["pointing"].shape = (3, 1)

        if k_len > 0:
            p_len = params["pointing"].shape[1]
            theta = np.empty((k_len, p_len), dtype=np.float64)
            for ind in range(p_len):
                theta[:, ind] = coordinates.vector_angle(
                    params["pointing"][:, ind],
                    k,
                    degrees=False,
                )
        else:
            theta = coordinates.vector_angle(params["pointing"], k, degrees=False)
            if len(params["pointing"].shape) == 1:
                theta = np.array([theta], dtype=np.float64)

        lam = scipy.constants.c / params["frequency"]
        k_n = 2.0 * np.pi / lam
        alph = (
            k_n[bcast["frequency"]]
            * params["radius"][bcast["radius"]]
            * np.sin(theta)[bcast["pointing"]]
        )
        if k_len > 0:
            alph = np.transpose(alph, tuple(range(1, len(G.shape))) + (0,))
        jn_val = scipy.special.jn(1, alph)

        inds_ = alph < 1e-9
        not_inds_ = np.logical_not(inds_)
        G[inds_] = self.I0
        G[not_inds_] = self.I0 * ((2.0 * jn_val[not_inds_] / alph[not_inds_])) ** 2.0

        return G
