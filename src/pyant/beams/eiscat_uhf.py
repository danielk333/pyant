#!/usr/bin/env python

"""

"""
import copy

import numpy as np
import scipy.interpolate

from ..models import Measured
from .. import coordinates
from .beams import radar_beam_generator
from ..registry import Radars, Models
from .data import DATA

_data_file = DATA["eiscat_uhf_bp.txt"] if "eiscat_uhf_bp.txt" in DATA else None
if _data_file is not None:
    with _data_file.open("r") as stream:
        _eiscat_beam_data = np.genfromtxt(stream)
else:
    _eiscat_beam_data = None


class EISCAT_UHF(Measured):
    """Measured gain pattern of the EISCAT UHF radar [1]_.

    .. [1] (Personal communication) Vierinen, J.

    """

    def __init__(self, azimuth, elevation, frequency, **kwargs):
        super().__init__(azimuth, elevation, frequency, **kwargs)
        self.I0 = 10**4.81
        self.F0 = 930e6

        angle = _eiscat_beam_data[:, 0]
        gain = 10.0 ** (_eiscat_beam_data[:, 1] / 10.0)

        self.beam_function = scipy.interpolate.interp1d(np.abs(angle), gain)

    def copy(self):
        """Return a copy of the current instance."""
        return EISCAT_UHF(
            azimuth=copy.deepcopy(self.azimuth),
            elevation=copy.deepcopy(self.elevation),
            frequency=copy.deepcopy(self.frequency),
            degrees=self.degrees,
        )

    def gain(self, k, polarization=None, ind=None, **kwargs):
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
                    degrees=True,
                )
        else:
            if p_len == 1:
                params["pointing"] = params["pointing"].reshape(3)
            theta = coordinates.vector_angle(params["pointing"], k, degrees=True)
            if theta.size == G.size:
                if len(theta.shape) > 0:
                    theta.shape = G.shape
            else:
                theta = np.broadcast_to(theta, G.shape)

        G = self.I0 * self.beam_function(params["frequency"] / self.F0 * theta)
        return G


@radar_beam_generator(Radars.EISCAT_UHF, Models.Measured)
def generate_eiscat_uhf_measured():
    return EISCAT_UHF(
        azimuth=0,
        elevation=90.0,
        frequency=930e6,
        degrees=True,
    )
