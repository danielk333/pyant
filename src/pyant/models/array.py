#!/usr/bin/env python

from dataclasses import dataclass
from typing import Callable, ClassVar
from numpy.typing import NDArray

import scipy.constants
import numpy as np

from ..beam import Beam, get_and_validate_k_shape
from ..types import (
    NDArray_2,
    NDArray_3,
    NDArray_N,
    NDArray_M,
    NDArray_Mx2,
    NDArray_2xN,
    NDArray_3xN,
    NDArray_MxN,
    NDArray_MxM,
    NDArray_3xNxM,
    NDArray_Mx2xN,
    Parameters,
)

AntennaGain = Callable[[NDArray_3 | NDArray_3xN, NDArray_2], NDArray_2xN]


def default_element(k: NDArray_3 | NDArray_3xN, polarization: NDArray_2) -> NDArray_2xN:
    """Antenna element gain pattern, azimuthally symmetric dipole response."""
    ret = np.ones((2,), dtype=k.dtype)
    return ret[:, None] * k[2, :]


@dataclass
class ArrayParams(Parameters):
    """
    Parameters
    ----------
    pointing
        Pointing direction of the phased array
    frequency
        Frequency of the radar
    polarization
        polarization of the incoming wave as Jones vector (complex valued)
    """

    pointing: NDArray_3xN | NDArray_3
    frequency: NDArray_N | float
    polarization: NDArray_2xN | NDArray_2

    pointing_shape: ClassVar[tuple[int, ...]] = (3,)
    frequency_shape: ClassVar[None] = None
    polarization_shape: ClassVar[tuple[int, ...]] = (2,)


class Array(Beam[ArrayParams]):
    """Gain pattern of an antenna array radar receiving/transmitting plane
    waves, i.e in the far field approximation regime. Assumes the same
    antenna is used throughout the array.

    Antennas can be combined into a single channel or multiple depending on
    the shape of the input `antenna` ndarray.

    Parameters
    ----------
    antennas
        `(3, n)` or `(3, n, m)` numpy array of antenna spatial positions,
        where `n` is the number of antennas and `m` is the number of sub-arrays.
        *This is not the same arrangement as the internal antennas variable*.
    scaling
        Scaling parameter for the output gain, can be interpreted as an
        antenna element scalar gain.
    polarization
        The Jones vector of the assumed polarization used when calculating
        the gain. Default is Left-hand circular polarized.

    Attributes
    ----------
    channels
        Number of sub-arrays the antenna array has, i.e the number of channels.

    """

    def __init__(
        self,
        antennas: NDArray_3xNxM,
        mutual_coupling_matrix: NDArray_MxM | None = None,
        antenna_element: AntennaGain = default_element,
        polarization: NDArray_2 = np.array([1, 1j]) / np.sqrt(2),
    ):
        super().__init__()
        if isinstance(antennas, list):
            for arr in antennas:
                assert arr.shape[0] == 3
                assert len(arr.shape) == 2
        else:
            assert len(antennas.shape) == 3
            assert antennas.shape[0] == 3
            assert isinstance(antennas, np.ndarray)

        if not np.all(np.iscomplex(polarization)):
            polarization = polarization.astype(np.complex128)

        self.mutual_coupling_matrix = mutual_coupling_matrix
        self.antennas = antennas
        self.polarization = polarization
        self.antenna_element = antenna_element

    def copy(self):
        """Return a copy of the current instance."""
        mem = self.mutual_coupling_matrix
        if mem is not None:
            mem = self.mutual_coupling_matrix.copy()
        if isinstance(self.antennas, list):
            antennas = [x.copy() for x in self.antennas]
        else:
            antennas = self.antennas.copy()
        return Array(
            antennas=antennas,
            polarization=self.polarization.copy(),
            mutual_coupling_matrix=mem,
            antenna_element=self.antenna_element,
        )

    @property
    def antenna_number(self) -> int:
        """Total number of antennas in the array."""
        if isinstance(self.antennas, list):
            return np.sum(x.shape[1] for x in self.antennas)
        else:
            return self.antennas.shape[1] * self.antennas.shape[2]

    @property
    def channels(self) -> int:
        """Number of channels returned by complex output."""
        if isinstance(self.antennas, list):
            return len(self.antennas)
        else:
            return self.antennas.shape[2]

    def gain(self, k: NDArray_3xN | NDArray_3, parameters: ArrayParams) -> NDArray_N | float:
        """Gain of the antenna array."""
        g = self.channel_signals(k, parameters)
        g = np.sum(g, axis=0)  # coherent intergeneration over channels
        return np.abs(g)

    def channel_signals(
        self, k: NDArray_3xN | NDArray_3, parameters: ArrayParams
    ) -> NDArray_MxN | NDArray_M:
        """Complex voltage output signals after summation of antennas and polarization.

        Returns
        -------
            `(c,num_k)` ndarray where `c` is the number of channels
            requested and `num_k` is the number of input wave vectors.
            If `num_k = 1` the returned ndarray is `(c,)`.
        """
        psi = self.signals(k, parameters, channels=None)

        lin_pol_check = np.abs(self.polarization) < 1e-6
        if not np.any(lin_pol_check):
            pol_comp = self.polarization[0] * self.polarization[1].conj()
            pol_comp = pol_comp.conj() / np.abs(pol_comp)
            psi[:, 0, ...] *= pol_comp  # align polarizations

        psi_out = np.sum(psi, axis=1)  # coherent intergeneration over polarization
        return psi_out

    def signals(
        self,
        k: NDArray_3 | NDArray_3xN,
        parameters: ArrayParams,
        channels: list[int] | None = None,
    ) -> NDArray_Mx2xN | NDArray_Mx2:
        """Complex voltage output signals after summation of antennas.

        Returns
        -------
            `(c,2,n)` ndarray where `c` is the number of channels
            requested, `2` are the two polarization axis of the Jones vector
            and `n` is the number of input wave vectors (i.e. a `(3,n)` matrix).
            If `n = 0`, i.e. the size is `(3,)`, the returned ndarray is
            `(c,2)`. If the parameters size is `n`, the return is `(c,2,n)`
            if k-vector size is 0, otherwise both has to be `n`.
        """
        size = parameters.size()
        k_len = get_and_validate_k_shape(size, k)
        if k_len == 0:
            return np.empty((self.channels, 0), dtype=np.complex128)
        scalar_output = size is None and k_len is None

        inds: NDArray[np.int64] = np.arange(self.channels, dtype=np.int64)
        if channels is not None:
            inds = inds[channels]

        p = parameters.pointing
        pol = parameters.polarization
        chan_num = len(inds)

        k = k / np.linalg.norm(k, axis=0)
        if scalar_output:
            psi = np.zeros((chan_num, 2, 1), dtype=np.complex128)
            k = k.reshape((3, 1))
            p = p.reshape((3, 1))
            pol = pol.reshape((2, 1))
        elif size is None and k_len is not None:
            psi = np.zeros((chan_num, 2, k_len), dtype=np.complex128)
            p = np.broadcast_to(p.reshape((3, 1)), (3, k_len))
            pol = np.broadcast_to(pol.reshape((2, 1)), (2, k_len))
        elif size is not None and k_len is not None:
            psi = np.zeros((chan_num, 2, k_len), dtype=np.complex128)
        elif size is not None and k_len is None:
            psi = np.zeros((chan_num, 2, size), dtype=np.complex128)
            k = np.broadcast_to(k.reshape((3, 1)), (3, size))

        wavelength = scipy.constants.c / parameters.frequency
        pol = pol.transpose((1, 0))

        # r in meters, divide by lambda
        for i in range(chan_num):
            if isinstance(self.antennas, list):
                grp = self.antennas[inds[i]][:, :]
            else:
                grp = self.antennas[:, :, inds[i]]

            kp = (k - p) / wavelength
            spat_wave = np.exp(1j * np.pi * 2.0 * np.sum(grp[:, :, None] * kp[:, None, :], axis=0))

            # broadcast to polarizations (pol was transposed to match dimensions)
            subg_response = spat_wave[:, :, None] * pol[None, :, :]

            psi[i, :, ...] = subg_response.sum(axis=0).T

        # This is an approximation assuming that the summed response of the subgroup
        # can be representative of the mutual coupling
        # TODO: There are better Mutual coupling models, they should be implemented specifically
        # for the radar where they work when we need them, this is left here as a reminder of this
        if self.mutual_coupling_matrix is not None:
            psi[:, 0, ...] = self.mutual_coupling_matrix @ psi[:, 0, ...]
            psi[:, 1, ...] = self.mutual_coupling_matrix @ psi[:, 1, ...]

        ant_response = self.antenna_element(k, pol)

        psi[:, 0, ...] *= ant_response[None, 0, ...]
        psi[:, 1, ...] *= ant_response[None, 1, ...]

        psi_out = psi.reshape(psi.shape[:-1]) if scalar_output else psi
        return psi_out
