#!/usr/bin/env python
import copy
from numpy.typing import NDArray
from typing import Callable
import numpy as np
import scipy.constants
import scipy.special

from ..beam import Beam
from ..types import (
    NDArray_2,
    NDArray_3,
    NDArray_N,
    NDArray_2xN,
    NDArray_3xN,
    NDArray_MxM,
    NDArray_3xNxM,
)

AntennaGain = Callable[[NDArray_3 | NDArray_3xN, NDArray_2], NDArray_2xN]


def default_element(k: NDArray_3 | NDArray_3xN, polarization: NDArray_2) -> NDArray_2xN:
    """Antenna element gain pattern, azimuthally symmetric dipole response."""
    ret = np.ones((2,), dtype=k.dtype)
    return ret[:, None] * k[2, :]


class Array(Beam):
    """Gain pattern of an antenna array radar receiving/transmitting plane
    waves, i.e in the far field approximation regime. Assumes the same
    antenna is used throughout the array.

    Antennas can be combined into a single channel or multiple depending on
    the shape of the input :code:`antenna` ndarray.

    Parameters
    ----------
    antennas : numpy.ndarray
        `(3, n)` or `(3, n, m)` numpy array of antenna spatial positions,
        where `n` is the number of antennas and `m` is the number of sub-arrays.
        *This is not the same arrangement as the internal antennas variable*.
    scaling : float
        Scaling parameter for the output gain, can be interpreted as an
        antenna element scalar gain.
    polarization : numpy.ndarray
        The Jones vector of the assumed polarization used when calculating
        the gain. Default is Left-hand circular polarized.

    Attributes
    ----------
    antennas : numpy.ndarray
        `(n, 3, m)` numpy array of antenna spatial positions,
        where `n` is the number of antennas and `m` is the number of sub-arrays.
    scaling : float
        Scaling parameter for the output gain, can be interpreted as an
        antenna element scalar gain.
    polarization : numpy.ndarray
        The Jones vector of the assumed polarization used when
        calculating the gain.
    channels : int
        Number of sub-arrays the antenna array has, i.e the number of channels.

    """

    def __init__(
        self,
        pointing: NDArray_3 | NDArray_3xN,
        frequency: NDArray_N,
        antennas: NDArray_3xNxM,
        mutual_coupling_matrix: NDArray_MxM | None = None,
        antenna_element: AntennaGain = default_element,
        polarization: NDArray_2 = np.array([1, 1j]) / np.sqrt(2),
        peak_gain: float = 1.0,
    ):
        super().__init__()
        self.parameters["pointing"] = pointing
        self.parameters_shape["pointing"] = (3,)
        self.parameters["frequency"] = frequency

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
        self.peak_gain = peak_gain
        self.polarization = polarization
        self.antenna_element = antenna_element
        self.validate_parameter_shapes()

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
            pointing=copy.deepcopy(self.parameters["pointing"]),
            frequency=copy.deepcopy(self.parameters["frequency"]),
            peak_gain=self.peak_gain,
            antennas=antennas,
            polarization=self.polarization.copy(),
            mutual_coupling_matrix=mem,
            antenna_element=self.antenna_element,
        )

    @property
    def channels(self) -> int:
        """Number of channels returned by complex output."""
        if isinstance(self.antennas, list):
            return len(self.antennas)
        else:
            return self.antennas.shape[2]

    def gain(self, k: NDArray_3 | NDArray_3xN, polarization: NDArray_2 | None = None):
        """Gain of the antenna array."""
        g = self.channel_signals(k, polarization=polarization)
        g = np.sum(g, axis=0)  # coherent intergeneration over channels
        return np.abs(g)

    def channel_signals(self, k: NDArray_3 | NDArray_3xN, polarization: NDArray_2 | None = None):
        """Complex voltage output signals after summation of antennas and polarization.

        Returns
        -------
        numpy.ndarray
            `(c,num_k)` ndarray where `c` is the number of channels
            requested and `num_k` is the number of input wave vectors.
            If `num_k = 1` the returned ndarray is `(c,)`.
        """
        if polarization is None:
            polarization = self.polarization
        elif not np.all(np.iscomplex(polarization)):
            polarization = polarization.astype(np.complex128)

        psi = self.signals(k, polarization, channels=None)

        lin_pol_check = np.abs(self.polarization) < 1e-6
        if not np.any(lin_pol_check):
            pol_comp = self.polarization[0] * self.polarization[1].conj()
            pol_comp = pol_comp.conj() / np.abs(pol_comp)
            psi[:, 0, ...] *= pol_comp  # align polarizations

        psi = np.sum(psi, axis=1)  # coherent intergeneration over polarization
        return psi

    def signals(
        self,
        k: NDArray_3 | NDArray_3xN,
        polarization: NDArray_2,
        channels: list[int] | None = None,
    ):
        """Complex voltage output signals after summation of antennas.

        Returns
        -------
        numpy.ndarray
            `(c,2,n)` ndarray where `c` is the number of channels
            requested, `2` are the two polarization axis of the Jones vector
            and `n` is the number of input wave vectors (i.e. a `(3,n)` matrix).
            If `n = 0`, i.e. the size is `(3,)`, the returned ndarray is
            `(c,2)`. If the parameters size is `n`, the return is `(c,2,n)`
            if k-vector size is 0, otherwise both has to be `n`.
        """
        k_len = self.validate_k_shape(k)
        size = self.size
        scalar_output = size == 0 and k_len == 0

        inds: NDArray[np.int64] = np.arange(self.channels, dtype=np.int64)
        if channels is not None:
            inds = inds[channels]

        p = self.parameters["pointing"]
        chan_num = len(inds)

        k = k / np.linalg.norm(k, axis=0)
        if size > 0 and k_len == 0:
            psi = np.zeros((chan_num, 2, size), dtype=np.complex128)
            k = np.broadcast_to(k.reshape((3, 1)), (3, size))
        elif size == 0 and k_len == 0:
            psi = np.zeros((chan_num, 2, 1), dtype=np.complex128)
            k = k.reshape((3, 1))
            p = p.reshape((3, 1))
        elif size == 0 and k_len > 0:
            psi = np.zeros((chan_num, 2, k_len), dtype=np.complex128)
            p = np.broadcast_to(p.reshape((3, 1)), (3, k_len))
        elif size > 0 and k_len > 0:
            psi = np.zeros((chan_num, 2, k_len), dtype=np.complex128)

        wavelength = scipy.constants.c / self.parameters["frequency"]

        # r in meters, divide by lambda
        for i in range(chan_num):
            if isinstance(self.antennas, list):
                grp = self.antennas[inds[i]][:, :]
            else:
                grp = self.antennas[:, :, inds[i]]

            kp = (k - p) / wavelength
            spat_wave = np.exp(1j * np.pi * 2.0 * np.sum(grp[:, :, None] * kp[:, None, :], axis=0))

            # broadcast to polarizations
            subg_response = spat_wave[:, :, None] * polarization[None, None, :]

            psi[i, :, ...] = subg_response.sum(axis=0).T

        # This is an approximation assuming that the summed response of the subgroup
        # can be representative of the mutual coupling
        # TODO: There are better Mutual coupling models, they should be implemented specifically
        # for the radar where they work when we need them, this is left here as a reminder of this
        if self.mutual_coupling_matrix is not None:
            psi[:, 0, ...] = self.mutual_coupling_matrix @ psi[:, 0, ...]
            psi[:, 1, ...] = self.mutual_coupling_matrix @ psi[:, 1, ...]

        ant_response = self.antenna_element(k, polarization) * self.peak_gain

        psi[:, 0, ...] *= ant_response[None, 0, ...]
        psi[:, 1, ...] *= ant_response[None, 1, ...]

        if scalar_output:
            psi = psi.reshape(psi.shape[:-1])
        return psi
