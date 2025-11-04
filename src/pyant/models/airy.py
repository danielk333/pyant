#!/usr/bin/env python

from dataclasses import dataclass
import json
from pathlib import Path
from typing import ClassVar, Type, TypeVar
import numpy as np
import scipy.constants
import scipy.special
import spacecoords.linalg as linalg

from ..beam import Beam, get_and_validate_k_shape
from ..types import NDArray_3, NDArray_3xN, NDArray_N, Parameters

T = TypeVar("T", bound="Airy")


@dataclass
class AiryParams(Parameters):
    """
    Parameters
    ----------
    pointing
        Pointing direction of the boresight
    frequency
        Frequency of the radar
    radius
        Radius in meters of the airy disk
    """

    pointing: NDArray_3xN | NDArray_3
    frequency: NDArray_N | float
    radius: NDArray_N | float

    pointing_shape: ClassVar[tuple[int, ...]] = (3,)
    frequency_shape: ClassVar[None] = None
    radius_shape: ClassVar[None] = None


class Airy(Beam[AiryParams]):
    """Airy disk gain model of a radar dish.

    Notes
    -----
    Singularities
        To avoid singularity at beam center, use
        :math:`\\lim_{x\\mapsto 0} \\frac{J_1(x)}{x} = \\frac{1}{2}` and a threshold.

    """

    def __init__(
        self,
        peak_gain: float = 1,
        zero_limit_eps: float = 1e-9,
    ):
        super().__init__()
        self.peak_gain = peak_gain
        self.zero_limit_eps = zero_limit_eps

    def to_json(self, path: Path):
        data = dict(
            peak_gain=self.peak_gain,
            zero_limit_eps=self.zero_limit_eps,
        )
        with open(path, "w") as fh:
            json.dump(data, fh)

    @classmethod
    def from_json(cls: Type[T], path: Path) -> T:
        with open(path, "r") as fh:
            data = json.load(fh)
        return cls(
            peak_gain=data["peak_gain"],
            zero_limit_eps=data["zero_limit_eps"],
        )

    def copy(self):
        """Return a copy of the current instance."""
        beam = Airy(
            peak_gain=self.peak_gain,
            zero_limit_eps=self.zero_limit_eps,
        )
        return beam

    def gain(self, k: NDArray_3xN | NDArray_3, parameters: AiryParams) -> NDArray_N | float:
        size = parameters.size()
        k_len = get_and_validate_k_shape(size, k)
        if k_len == 0:
            return np.empty((0,), dtype=k.dtype)

        scalar_output = size is None and k_len is None

        p = parameters.pointing
        # size of theta is always k_len or size or a scalar
        theta = linalg.vector_angle(p, k, degrees=False)

        lam = scipy.constants.c / parameters.frequency
        k_n = 2.0 * np.pi / lam
        radius = parameters.radius

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
