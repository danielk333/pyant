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
        p = self.get_parameters(ind, named=True)
        inds = self.ind_to_dict(ind)
        G = self._generate_gain_array(inds)

        lam = scipy.constants.c / p["frequency"]
        theta = coordinates.vector_angle(p["pointing"], k, degrees=False)

        k_n = 2.0 * np.pi / lam
        alph = np.outer(np.outer(k_n, p["radius"]), np.sin(theta))
        jn_val = scipy.special.jn(1, alph)

        inds_ = alph < 1e-9
        not_inds_ = np.logical_not(inds_)
        G[inds_] = self.I0
        G[not_inds_] = self.I0 * ((2.0 * jn_val[not_inds_] / alph[not_inds_])) ** 2.0

        return G
