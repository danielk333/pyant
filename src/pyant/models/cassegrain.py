#!/usr/bin/env python
import copy

import numpy as np
import scipy.special

from ..beam import Beam
from .. import coordinates


class Cassegrain(Beam):
    """Cassegrain gain model of a radar dish.

    :param float I0: Peak gain (linear scale) in the pointing direction.
    :param float a0: outer radius (main reflector)
    :param float a1: inner radius (subreflector)

    The gain pattern is expressed as the Fourier transform of an annular region
    with outer radius a0 and inner radius a1.  Values below the aperture plane
    or below the horizon should not be believed.
    """

    def __init__(self, azimuth, elevation, frequency, I0, a0, a1, **kwargs):
        super().__init__(azimuth, elevation, frequency, **kwargs)
        self.I0 = I0
        self.a1 = a1
        self.a0 = a0

    def copy(self):
        """Return a copy of the current instance."""
        return Cassegrain(
            azimuth=copy.deepcopy(self.azimuth),
            elevation=copy.deepcopy(self.elevation),
            frequency=copy.deepcopy(self.frequency),
            I0=copy.deepcopy(self.I0),
            a1=copy.deepcopy(self.a1),
            a0=copy.deepcopy(self.a0),
            degrees=self.degrees,
        )

    def gain(
        self, k, polarization=None, ind=None, vectorized_parameters=False, **kwargs
    ):
        if vectorized_parameters:
            raise NotImplementedError(
                "vectorized_parameters is not supported by Cassegrain"
            )

        pointing, frequency = self.get_parameters(
            ind, vectorized_parameters=vectorized_parameters, **kwargs
        )

        theta = coordinates.vector_angle(pointing, k, degrees=False)

        lam = scipy.constants.c / frequency

        eps = 1e-6

        if len(k.shape) == 1:
            theta = np.array([theta], dtype=k.dtype)

        G = np.empty((len(theta),), dtype=k.dtype)
        nz_idx_ = (
            np.pi * np.sin(theta) > 1e-9
        )  # pointings not close to zero off-axis angle
        nzth = theta[nz_idx_]

        a0, a1, I0 = self.a0, self.a1, self.I0

        G[:] = I0  # will overwrite for nonzero pointings below

        def scale(th):
            return lam / (np.pi * np.sin(th))

        def J1(x):
            return scipy.special.jn(1, x)

        def A(th):
            return (scale(th) / (a0**2 - a1**2)) ** 2

        def B(th):
            return (a0 * J1(a0 / scale(th)) - a1 * J1(a1 / scale(th))) ** 2

        G[nz_idx_] = self.I0 * (A(nzth) * B(nzth)) / (A(eps) * B(eps))

        if len(k.shape) == 1:
            G = G[0]

        return G
