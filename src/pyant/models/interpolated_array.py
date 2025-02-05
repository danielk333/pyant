import copy

import numpy as np
import scipy.interpolate

from ..beam import Beam
from .array import Array
from .. import coordinates

# import matplotlib.pyplot as plt


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


class InterpolatedArray(Beam):
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

    Parameters
    ----------
    scaling : float
        Scaling of the gain pattern to apply.

    Attributes
    ----------
    scaling : float
        Scaling of the gain pattern to apply.

    """

    def __init__(self, scaling=1.0, azimuth=0, elevation=0, **kwargs):
        super().__init__(azimuth=azimuth, elevation=elevation, frequency=np.nan, **kwargs)
        self.interpolated = None
        self.interpolated_channels = []
        self.interpolated_antenna = None
        self.scaling = scaling
        self.channels = None
        self.interp_dims = None

    def copy(self):
        """Return a copy of the current instance."""
        bm = InterpolatedArray(
            azimuth=copy.deepcopy(self.azimuth),
            elevation=copy.deepcopy(self.elevation),
            scaling=copy.deepcopy(self.scaling),
            degrees=self.degrees,
        )
        bm.frequency = self.frequency
        bm.pointing = self.pointing
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
                frequency=self.frequency,
                pointing=self.pointing,
                degrees=self.degrees,
                interp_dims=self.interp_dims,
            )
        )
        np.savez(fname, **datas)

    def load(self, fname):
        data = np.load(fname, allow_pickle=True)
        self.channels = data["channels"]
        self.frequency = data["frequency"]
        self.degrees = data["degrees"]
        self.pointing = data["pointing"]
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
        beam,
        ind=None,
        polarization=None,
        min_elevation=0.0,
        interpolate_channels=None,
        resolution=1000,
    ):
        """Generate an interpolated version of Array

        Parameters
        ----------
        channels : optional, list of int
            Index for which channels to save interpolations from.

        """
        assert isinstance(beam, Array), "Can only interpolate arrays"

        # Transfer meta-data and parameters
        self.channels = beam.channels

        if polarization is None:
            polarization = self.polarization
        elif not np.all(np.iscomplex(polarization)):
            polarization = polarization.astype(np.complex128)

        params, shape = beam.get_parameters(ind, named=True, max_vectors=0)
        self.frequency = params["frequency"]
        p = params["pointing"].reshape(3)
        sph = coordinates.cart_to_sph(p, degrees=beam.degrees)
        self.degrees = beam.degrees
        self.sph_point(sph[0], sph[1])

        wavelength = scipy.constants.c / params["frequency"]
        if len(wavelength.shape) > 0:
            wavelength = wavelength[0]
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

        if resolution[2] is None:
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
        G_subgrp = np.zeros((size,), dtype=np.float64)
        self.interpolated_channels = [None] * self.channels
        self.interpolated = None
        self.interpolated_antenna = None

        psi = np.zeros(
            (self.channels, size),
            dtype=np.complex128,
        )
        # r in meters, divide by lambda
        for i in range(self.channels):
            if isinstance(self.antennas, list):
                grp = beam.antennas[i][:, :].T
            else:
                grp = beam.antennas[:, :, i].T
            subg_response = plane_wave_compund(kp[:, inds], grp / wavelength)
            psi[i, inds] = subg_response.sum(axis=0).T

        ant_response = beam.antenna_element(k, polarization) * beam.scaling

        # broadcast over input polarization
        ant_response = ant_response[:, :] * polarization[:, None]

        # align according to receiving polarizations
        pol_comp = self.polarization[0] * self.polarization[1].conj()
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
            G_subgrp[inds] = np.abs(psi[i, inds])
            # coherent intergeneration over channels
            sum_psi[inds] += psi[i, inds]

            if interpolate_channels is None or i not in interpolate_channels:
                continue

            if resolution[2] is None:
                self.interpolated_channels[i] = scipy.interpolate.RegularGridInterpolator(
                    (kpx, kpy),
                    G_subgrp.reshape(*resolution[:2]),
                    bounds_error=False,
                )
            else:
                self.interpolated_channels[i] = scipy.interpolate.RegularGridInterpolator(
                    (kpx, kpy, kpz),
                    G_subgrp.reshape(*resolution),
                    bounds_error=False,
                )
        G = np.abs(sum_psi).astype(np.float64)
        if resolution[2] is None:
            self.interpolated = scipy.interpolate.RegularGridInterpolator(
                (kpx, kpy),
                G.reshape(*resolution[:2]),
                bounds_error=False,
            )
        else:
            self.interpolated = scipy.interpolate.RegularGridInterpolator(
                (kpx, kpy, kpz),
                G.reshape(*resolution),
                bounds_error=False,
            )

    def channel_gain(self, k, channels=None, ind=None, **kwargs):
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
        k_len = k.shape[1] if len(k.shape) == 2 else 1
        assert len(k.shape) <= 2, "'k' can only be vectorized with one additional axis"
        params, shape = self.get_parameters(ind, named=True, max_vectors=0)
        if k_len == 1:
            k = k.reshape(3, 1)
        p = params["pointing"].reshape(3)

        kn = k / np.linalg.norm(k, axis=0)
        kpn = kn - p[:, None]

        if channels is None:
            channels = np.arange(self.channels)
        gains = np.full((len(channels), k_len), np.nan, dtype=np.float64)

        for ind, ci in enumerate(channels):
            if self.interpolated_channels[ci] is None:
                continue
            if self.interp_dims == 3:
                gains[ind, :] = self.interpolated_channels[ci](kpn[:3, ...].T)
            else:
                gains[ind, :] = self.interpolated_channels[ci](kpn[:2, ...].T)

        ant_response = self.interpolated_antenna(kn[:2, ...].T)

        gains *= ant_response[None, :]
        if len(k.shape) == 1:
            gains = gains.reshape(gains.shape[:-1])

        return gains

    def gain(self, k, ind=None, polarization=None, **kwargs):
        """Gain of the antenna array."""
        k_len = k.shape[1] if len(k.shape) == 2 else 1
        assert len(k.shape) <= 2, "'k' can only be vectorized with one additional axis"
        params, shape = self.get_parameters(ind, named=True, max_vectors=0)
        if k_len == 1:
            k = k.reshape(3, 1)
        p = params["pointing"].reshape(3)

        kn = k / np.linalg.norm(k, axis=0)
        kpn = kn - p[:, None]

        if self.interp_dims == 3:
            G = self.interpolated(kpn[:3, ...].T)
        else:
            G = self.interpolated(kpn[:2, ...].T)

        ant_response = self.interpolated_antenna(kn[:2, ...].T)

        G *= ant_response
        if len(k.shape) == 1:
            G = G.reshape(G.shape[:-1])
        return G
