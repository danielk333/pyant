#!/usr/bin/env python

from dataclasses import dataclass
import json
from pathlib import Path
from typing import ClassVar, Type, TypeVar
import numpy as np
import scipy.special
import spacecoords.linalg as linalg

from ..beam import Beam, get_and_validate_k_shape
from ..types import NDArray_3, NDArray_3xN, NDArray_N, Parameters

T = TypeVar("T", bound="Cassegrain")


def calculate_cassegrain_AB(
    theta: NDArray_N | float, lam: NDArray_N | float, a0: NDArray_N | float, a1: NDArray_N | float
):
    """Calculates the A and B parameters in the Cassegrain gain model"""
    scale = lam / (np.pi * np.sin(theta))
    A = (scale / (a0**2 - a1**2)) ** 2
    ba0 = a0 * scipy.special.j1(a0 / scale)
    ba1 = a1 * scipy.special.j1(a1 / scale)
    B = (ba0 - ba1) ** 2

    return A, B


@dataclass
class CassegrainParams(Parameters):
    """
    Parameters
    ----------
    pointing
        Pointing direction of the boresight
    frequency
        Frequency of the radar
    outer_radius
        Radius of main reflector
    inner_radius
        Radius of sub reflector
    """

    pointing: NDArray_3xN | NDArray_3
    frequency: NDArray_N | float
    outer_radius: NDArray_N | float
    inner_radius: NDArray_N | float

    pointing_shape: ClassVar[tuple[int, ...]] = (3,)
    frequency_shape: ClassVar[None] = None
    outer_radius_shape: ClassVar[None] = None
    inner_radius_shape: ClassVar[None] = None


class Cassegrain(Beam[CassegrainParams]):
    """Cassegrain gain model of a radar dish.

    Parameters
    ----------
    peak_gain
        Peak gain (linear scale) in the pointing direction.
    min_off_axis
        Minimum off axis angle where we instead use the limit value
    eps
        ??

    Notes
    -----
    Derivation
        The gain pattern is expressed as the Fourier transform of an annular
        region with outer radius a0 and inner radius a1.  Values below the
        aperture plane or below the horizon should not be believed.
    """

    def __init__(
        self,
        peak_gain: float = 1,
        min_off_axis: float = 1e-9,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.min_off_axis = min_off_axis
        self.peak_gain = peak_gain

    def to_json(self, path: Path):
        data = dict(
            peak_gain=self.peak_gain,
            min_off_axis=self.min_off_axis,
            eps=self.eps,
        )
        with open(path, "w") as fh:
            json.dump(data, fh)

    @classmethod
    def from_json(cls: Type[T], path: Path) -> T:
        with open(path, "r") as fh:
            data = json.load(fh)
        return cls(
            peak_gain=data["peak_gain"],
            min_off_axis=data["min_off_axis"],
            eps=data["eps"],
        )

    def copy(self):
        """Return a copy of the current instance."""
        beam = Cassegrain(
            peak_gain=self.peak_gain,
            min_off_axis=self.min_off_axis,
            eps=self.eps,
        )
        return beam

    def gain(self, k: NDArray_3xN | NDArray_3, parameters: CassegrainParams) -> NDArray_N | float:
        size = parameters.size()
        k_len = get_and_validate_k_shape(size, k)
        if k_len == 0:
            return np.empty((0,), dtype=k.dtype)
        scalar_output = size is None and k_len is None

        p = parameters.pointing
        theta = linalg.vector_angle(p, k, degrees=False)

        lam = scipy.constants.c / parameters.frequency
        a0 = parameters.outer_radius
        a1 = parameters.inner_radius

        theta_arr = np.array([theta]) if isinstance(theta, float) else theta

        # pointings not close to zero off-axis angle
        inds = np.pi * np.sin(theta) > self.min_off_axis

        if size is not None:
            # If the size is not None we know these are all numpy arrays
            a0, a1, lam = a0[inds], a1[inds], lam[inds]  # type: ignore

        g = np.empty(theta_arr.shape, dtype=np.float64)

        A, B = calculate_cassegrain_AB(theta_arr[inds], lam, a0, a1)
        A_eps, B_eps = calculate_cassegrain_AB(self.eps, lam, a0, a1)

        g[np.logical_not(inds)] = self.peak_gain
        g[inds] = self.peak_gain * (A * B) / (A_eps * B_eps)

        if scalar_output:
            g = g[0]
        return g
