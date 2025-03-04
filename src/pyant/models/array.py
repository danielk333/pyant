#!/usr/bin/env python
import copy

import numpy as np
import scipy.constants
import scipy.special

from ..beam import Beam


def default_element(k, polarization):
    """Antenna element gain pattern, azimuthally symmetric dipole response."""
    ret = np.ones((2,), dtype=k.dtype)
    return ret[:, None] * k[2, :]


def plane_wave(k, r, p, J):
    """The complex plane wave function.

    Parameters
    ----------
    k : numpy.ndarray
        Wave-vectors (wave propagation directions)
    r : numpy.ndarray
        Spatial locations (Antenna positions in space)
    p : numpy.ndarray
        Beam-forming direction by phase offsets
        (antenna array "pointing" direction)
    J : numpy.ndarray
        Polarization given as a Jones vector
    """

    # in this, rows are antennas and columns are wave directions
    spat_wave = np.exp(1j * np.pi * 2.0 * np.dot(r, k - p))

    # broadcast to polarizations
    wave = spat_wave[..., None] * J[None, None, :]

    return wave


class Array(Beam):
    """Gain pattern of an antenna array radar receiving/transmitting plane
    waves, i.e in the far field approximation regime. Assumes the same
    antenna is used throughout the array.

    Antennas can be combined into a single channel or multiple depending on
    the shape of the input :code:`antenna` ndarray.

    Parameters
    ----------
    antennas : numpy.ndarray
        `(3, n)` or `(3, n, c)` numpy array of antenna spatial positions,
        where `n` is the number of antennas and `c` is the number of sub-arrays.
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
        `(n, 3, c)` numpy array of antenna spatial positions,
        where `n` is the number of antennas and `c` is the number of sub-arrays.
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
        azimuth,
        elevation,
        frequency,
        antennas,
        mutual_coupling_matrix=None,
        antenna_element=default_element,
        polarization=np.array([1, 1j]) / np.sqrt(2),
        scaling=1.0,
        **kwargs
    ):
        super().__init__(azimuth, elevation, frequency, **kwargs)
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
        self.scaling = scaling
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
            frequency=copy.deepcopy(self.frequency),
            azimuth=copy.deepcopy(self.azimuth),
            elevation=copy.deepcopy(self.elevation),
            degrees=self.degrees,
            antennas=antennas,
            scaling=copy.deepcopy(self.scaling),
            polarization=self.polarization.copy(),
            mutual_coupling_matrix=mem,
            antenna_element=self.antenna_element,
        )

    @property
    def channels(self):
        """Number of channels returned by complex output."""
        if isinstance(self.antennas, list):
            return len(self.antennas)
        else:
            return self.antennas.shape[2]

    def gain(self, k, ind=None, polarization=None, **kwargs):
        """Gain of the antenna array."""
        G = self.channel_signals(k, ind=ind, polarization=polarization, **kwargs)
        G = np.sum(G, axis=0)  # coherent intergeneration over channels
        return np.abs(G)

    def channel_signals(self, k, ind=None, polarization=None, **kwargs):
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

        psi = self.signals(k, polarization, channels=None, ind=ind, **kwargs)

        lin_pol_check = np.abs(self.polarization) < 1e-6
        if not np.any(lin_pol_check):
            pol_comp = self.polarization[0] * self.polarization[1].conj()
            pol_comp = pol_comp.conj() / np.abs(pol_comp)
            psi[:, 0, ...] *= pol_comp  # align polarizations

        psi = np.sum(psi, axis=1)  # coherent intergeneration over polarization
        return psi

    def signals(self, k, polarization, ind=None, channels=None, **kwargs):
        """Complex voltage output signals after summation of antennas.

        Returns
        -------
        numpy.ndarray
            `(c,2,num_k)` ndarray where `c` is the number of channels
            requested, `2` are the two polarization axis of the Jones vector
            and `num_k` is the number of input wave vectors. If `num_k = 1`
            the returned ndarray is `(c,2)`.
        """
        k_len = k.shape[1] if len(k.shape) == 2 else 0
        assert len(k.shape) <= 2, "'k' can only be vectorized with one additional axis"

        params, shape = self.get_parameters(ind, named=True, max_vectors=0)

        inds = np.arange(self.channels, dtype=np.int64)
        if channels is not None:
            inds = inds[channels]

        p = params["pointing"].reshape(3, 1)

        chan_num = len(inds)

        k_ = k / np.linalg.norm(k, axis=0)
        if k_len == 0:
            psi = np.zeros((chan_num, 2, 1), dtype=np.complex128)
            k_ = k_.reshape(3, 1)
        else:
            psi = np.zeros(
                (chan_num, 2, k_len),
                dtype=np.complex128,
            )
            p = np.repeat(p, k_len, axis=1)

        wavelength = scipy.constants.c / params["frequency"]
        if len(wavelength.shape) > 0:
            wavelength = wavelength[0]

        # r in meters, divide by lambda
        for i in range(chan_num):
            if isinstance(self.antennas, list):
                grp = self.antennas[inds[i]][:, :].T
            else:
                grp = self.antennas[:, :, inds[i]].T
            subg_response = plane_wave(k_, grp / wavelength, p, polarization)
            psi[i, :, ...] = subg_response.sum(axis=0).T

        # This is an approximation assuming that the summed response of the subgroup
        # can be representative of the mutual coupling
        if self.mutual_coupling_matrix is not None:
            psi[:, 0, ...] = self.mutual_coupling_matrix @ psi[:, 0, ...]
            psi[:, 1, ...] = self.mutual_coupling_matrix @ psi[:, 1, ...]

        ant_response = self.antenna_element(k_, polarization) * self.scaling

        psi[:, 0, ...] *= ant_response[None, 0, ...]
        psi[:, 1, ...] *= ant_response[None, 1, ...]

        if len(k.shape) == 1:
            psi = psi.reshape(psi.shape[:-1])

        return psi
