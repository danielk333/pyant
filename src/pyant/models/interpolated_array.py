#!/usr/bin/env python

from dataclasses import dataclass
from typing import ClassVar
import copy

import numpy as np
import scipy.interpolate

from ..beam import Beam, get_and_validate_k_shape
from . import array
from ..types import (
    NDArray_3,
    NDArray_3xN,
    NDArray_N,
    NDArray_M,
    NDArray_MxN,
    Parameters,
)


def plane_wave_compund(kp, r):
    """The complex plane wave function.

    Parameters
    ----------
    kp : numpy.ndarray
        Wave-vectors minus pointing direction (wave propagation directions)
    r : numpy.ndarray
        Spatial locations normalized by wavelength (Antenna positions in space)
    """
    # in this, rows are antennas and columns are wave directions
    wave = np.exp(1j * np.pi * 2.0 * np.dot(r, kp))
    return wave


@dataclass
class InterpolatedArrayParams(Parameters):
    """
    Parameters
    ----------
    pointing
        Pointing direction of the array
    """

    pointing: NDArray_3xN | NDArray_3

    pointing_shape: ClassVar[tuple[int, ...]] = (3,)


class InterpolatedArray(Beam[InterpolatedArrayParams]):
    """Interpolated gain pattern of array and each subgroup.
    DOES NOT REPRODUCE COMPLEX VOLTAGES.

    Assumes that the gain as a function of the incoming/outgoing wave vector
    and pointing vector can be substituted by
    $\\mathbf{q} = \\mathbf{k} - \\mathbf{p}$, turning the 4D problem
    into a 3D problem. Because of how the plane wave equation scales with
    wavelength one can *not* use different frequencies with the same
    interpolation. As complex voltages are not interpolated, one cannot
    set polarization or change the antenna element. Mutual coupling is not
    supported. Only scaling can be set freely. If the pointing is fixed
    the problem is again a 2D one and a 2D interpolation can be made.

    """

    def __init__(self):
        super().__init__()
        self.channels = None
        self.interp_dims = None
        self.interpolated = None
        self.interpolated_antenna = None
        self.interpolated_channels = None

    def copy(self):
        """Return a copy of the current instance."""
        bm = InterpolatedArray()
        bm.channels = self.channels
        bm.interp_dims = self.interp_dims
        bm.interpolated = copy.deepcopy(self.interpolated)
        bm.interpolated_antenna = copy.deepcopy(self.interpolated_antenna)
        bm.interpolated_channels = [copy.deepcopy(x) for x in self.interpolated_channels]
        return bm

    def save(self, fname):
        datas = {
            f"interp_chan{ind}": self.interpolated_channels[ind]
            for ind in range(self.channels)
            if self.interpolated_channels[ind] is not None
        }
        datas["interpolated"] = self.interpolated
        datas["interpolated_antenna"] = self.interpolated_antenna
        datas.update(
            dict(
                channels=np.int64(self.channels),
                interp_dims=self.interp_dims,
            )
        )
        np.savez(fname, **datas)

    def load(self, fname):
        data = np.load(fname, allow_pickle=True)
        self.channels = data["channels"]
        self.interp_dims = data["interp_dims"]

        self.interpolated_channels = [None] * self.channels
        for ind in range(self.channels):
            if f"interp_chan{ind}" not in data:
                continue
            self.interpolated_channels[ind] = data[f"interp_chan{ind}"].item()
        self.interpolated = data["interpolated"].item()
        self.interpolated_antenna = data["interpolated_antenna"].item()

    def generate_interpolation(
        self,
        beam: array.Array,
        parameters: array.ArrayParams,
        min_elevation: float = 0.0,
        interpolate_channels: list[int] | None = None,
        resolution: tuple[int, int] | tuple[int, int, int] = (1000, 1000),
    ):
        """Generate an interpolated version of Array

        Parameters
        ----------
        channels
            Index for which channels to save interpolations from.

        """
        if not isinstance(beam, array.Array):
            raise TypeError(f"Can only interpolate Array, not '{type(beam)}'")

        if parameters.size is None:
            raise ValueError(
                "Can only plot beam with scalar parameters -"
                f"dont know which of the {parameters.size} options to pick"
            )

        if "pointing" not in parameters.keys:
            p = np.array([0, 0, 1], dtype=np.float64)
        else:
            p = parameters.pointing  # type: ignore

        # Transfer meta-data and parameters
        self.channels = beam.channels
        pol = parameters.polarization

        wavelength = scipy.constants.c / parameters.frequency
        cmin = np.cos(np.radians(min_elevation))

        size_k = np.prod(resolution[:2])
        k = np.empty((3, size_k), dtype=np.float64)
        kx = np.linspace(-cmin, cmin, num=resolution[0])
        ky = np.linspace(-cmin, cmin, num=resolution[1])
        xv, yv = np.meshgrid(kx, ky, sparse=False, indexing="ij")
        k[0, :] = xv.reshape(1, size_k)
        k[1, :] = yv.reshape(1, size_k)
        xy2 = k[0, :] ** 2 + k[1, :] ** 2
        inds = xy2 <= cmin
        k[2, inds] = np.sqrt(1 - xy2[inds])
        k[2, np.logical_not(inds)] = 0

        if len(resolution) == 2:
            size = size_k
            kp = k[:, :] - p[:, None]
            kpx = kx - p[0]
            kpy = ky - p[1]
            self.interp_dims = 2
        else:
            size = np.prod(resolution)
            kpx = np.linspace(-2.0, 2.0, num=resolution[0])
            kpy = np.linspace(-2.0, 2.0, num=resolution[1])
            kpz = np.linspace(-2.0, 2.0, num=resolution[2])
            xv, yv, zv = np.meshgrid(kpx, kpy, kpz, sparse=False, indexing="ij")
            kp = np.empty((3, size), dtype=np.float64)
            kp[0, :] = xv.reshape(1, size)
            kp[1, :] = yv.reshape(1, size)
            kp[2, :] = zv.reshape(1, size)

            xy2 = kp[0, :] ** 2 + kp[1, :] ** 2

            norm = np.linalg.norm(kp, axis=0)
            inds = norm <= 2.0
            self.interp_dims = 3

        sum_psi = np.zeros((size,), dtype=np.complex128)
        g_subgrp = np.zeros((size,), dtype=np.float64)
        self.interpolated_channels = [None] * self.channels
        self.interpolated = None
        self.interpolated_antenna = None

        psi = np.zeros(
            (self.channels, size),
            dtype=np.complex128,
        )
        # r in meters, divide by lambda
        for i in range(self.channels):
            if isinstance(beam.antennas, list):
                grp = beam.antennas[i][:, :].T
            else:
                grp = beam.antennas[:, :, i].T
            subg_response = plane_wave_compund(kp[:, inds], grp / wavelength)
            psi[i, inds] = subg_response.sum(axis=0).T

        ant_response = beam.antenna_element(k, pol)

        # broadcast over input polarization
        ant_response = ant_response[:, :] * pol[:, None]

        # align according to receiving polarizations
        lin_pol_check = np.abs(beam.polarization) < 1e-6
        if not np.any(lin_pol_check):
            pol_comp = beam.polarization[0] * beam.polarization[1].conj()
            pol_comp = pol_comp.conj() / np.abs(pol_comp)
            ant_response[0, ...] *= pol_comp

        # coherent intergeneration over polarization
        ant_response = np.abs(np.sum(ant_response, axis=0)).astype(np.float64)
        self.interpolated_antenna = scipy.interpolate.RegularGridInterpolator(
            (kx, ky),
            ant_response.reshape(*resolution[:2]),
            bounds_error=False,
        )

        for i in range(self.channels):
            g_subgrp[inds] = np.abs(psi[i, inds])
            # coherent intergeneration over channels
            sum_psi[inds] += psi[i, inds]

            if interpolate_channels is None or i not in interpolate_channels:
                continue

            if len(resolution) == 2:
                self.interpolated_channels[i] = scipy.interpolate.RegularGridInterpolator(
                    (kpx, kpy),
                    g_subgrp.reshape(*resolution[:2]),
                    bounds_error=False,
                )
            else:
                self.interpolated_channels[i] = scipy.interpolate.RegularGridInterpolator(
                    (kpx, kpy, kpz),
                    g_subgrp.reshape(*resolution),
                    bounds_error=False,
                )
        g = np.abs(sum_psi).astype(np.float64)
        if len(resolution) == 2:
            self.interpolated = scipy.interpolate.RegularGridInterpolator(
                (kpx, kpy),
                g.reshape(*resolution[:2]),
                bounds_error=False,
            )
        else:
            self.interpolated = scipy.interpolate.RegularGridInterpolator(
                (kpx, kpy, kpz),
                g.reshape(*resolution),
                bounds_error=False,
            )

    def channel_gain(
        self, k: NDArray_3xN | NDArray_3, parameters: InterpolatedArrayParams, channels=None
    ) -> NDArray_MxN | NDArray_M:
        """Interpolated gain of each channel.

        Parameters
        ----------
        channels : optional, list of int
            Index for which channels to get gains from. If none, get all available.

        Returns
        -------
        numpy.ndarray
            `(c,num_k)` ndarray where `c` is the number of channels
            requested and `num_k` is the number of input wave vectors.
            If `num_k = 1` the returned ndarray is `(c,)`.
        """
        size = parameters.size()
        k_len = get_and_validate_k_shape(size, k)
        if k_len == 0:
            return np.empty((self.channels, 0), dtype=k.dtype)

        p = parameters.pointing

        kn = k / np.linalg.norm(k, axis=0)
        if size is not None and k_len is not None:
            kpn = kn - p
        elif size is not None and k_len is None:
            kpn = kn[:, None] - p
        elif size is None and k_len is not None:
            kpn = kn - p[:, None]
        elif size is None and k_len is None:
            kpn = kn - p
        g_size = kpn.shape[1] if len(kpn.shape) > 1 else 1

        if channels is None:
            channels = np.arange(self.channels)
        gains = np.full((len(channels), g_size), np.nan, dtype=np.float64)

        for ind, ci in enumerate(channels):
            if self.interpolated_channels[ci] is None:
                continue
            if self.interp_dims == 3:
                gains[ind, :] = self.interpolated_channels[ci](kpn[:3, ...].T)
            else:
                gains[ind, :] = self.interpolated_channels[ci](kpn[:2, ...].T)

        ant_response = self.interpolated_antenna(kn[:2, ...].T)

        gains = gains * ant_response[None, :]
        if len(kpn.shape) == 1:
            gains = gains.reshape(gains.shape[:-1])

        return gains

    def gain(
        self,
        k: NDArray_3xN | NDArray_3,
        parameters: InterpolatedArrayParams,
    ) -> NDArray_N | float:
        """Gain of the antenna array."""
        size = parameters.size()
        k_len = get_and_validate_k_shape(size, k)
        if k_len == 0:
            return np.empty((self.channels, 0), dtype=k.dtype)
        scalar_output = size is None and k_len is None

        p = parameters.pointing

        kn = k / np.linalg.norm(k, axis=0)
        if size is not None and k_len is not None:
            kpn = kn - p
        elif size is not None and k_len is None:
            kpn = kn[:, None] - p
            kn = np.broadcast_to(kn.reshape((3, 1)), (3, size))
        elif size is None and k_len is not None:
            kpn = kn - p[:, None]
        elif scalar_output:
            kpn = kn - p

        if self.interp_dims == 3:
            g = self.interpolated(kpn[:3, ...].T)
        else:
            g = self.interpolated(kpn[:2, ...].T)

        ant_response = self.interpolated_antenna(kn[:2, ...].T)

        g = g * ant_response
        if scalar_output:
            g = g[0]
        return g
