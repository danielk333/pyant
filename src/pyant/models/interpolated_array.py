import copy

import numpy as np
import scipy.interpolate

from .interpolated import Interpolated
from .array import Array


def plane_wave_compund(kp, r):
    """The complex plane wave function.

    Parameters
    ----------
    k : numpy.ndarray
        Wave-vectors minus pointing direction (wave propagation directions)
    r : numpy.ndarray
        Spatial locations normalized by wavelength (Antenna positions in space)
    """

    # in this, rows are antennas and columns are wave directions
    spat_wave = np.exp(1j * np.pi * 2.0 * np.dot(r, kp))

    return spat_wave


class InterpolatedArray(Interpolated):
    """Interpolated gain pattern. Assumes that the gain as a function of the
    incoming/outgoing wave vector and pointing vector can be substituted by
    :math:`\\mathbf{q} = \\mathbf{k} - \\mathbf{p}`, turning the 4D problem
    into a 3D problem. Because of how the plane wave equation scales with
    wavelength one can *not* use different frequencies with the same
    interpolation. However, scaling and polarization can be set freely.

    Parameters
    ----------
    scaling : float
        Scaling of the gain pattern to apply.

    Attributes
    ----------
    scaling : float
        Scaling of the gain pattern to apply.

    """

    def __init__(
        self,
        azimuth,
        elevation,
        frequency,
        polarization=np.array([1, 1j]) / np.sqrt(2),
        scaling=1.0,
        **kwargs
    ):
        super().__init__(azimuth, elevation, frequency, **kwargs)
        self.interpolated = None
        self.scaling = scaling
        self.polarization = polarization
        self.channels = None
        self.mutual_coupling_matrix = None
        self.antennas = None

    def antenna_element(self, k, polarization):
        """Antenna element gain pattern, azimuthally symmetric dipole response."""
        ret = np.ones(polarization.shape, dtype=k.dtype)
        return ret[:, None] * k[2, :] * self.scaling

    def copy(self):
        """Return a copy of the current instance."""
        bm = Interpolated(
            azimuth=copy.deepcopy(self.azimuth),
            elevation=copy.deepcopy(self.elevation),
            frequency=copy.deepcopy(self.frequency),
            scaling=copy.deepcopy(self.scaling),
            degrees=self.degrees,
        )
        bm.interpolated = self.interpolated
        return bm

    def save(self, fname):
        interps = {f"interp{ind}": self.interpolated[ind] for ind in range(self.channels)}
        np.savez(
            fname,
            mutual_coupling_matrix=self.mutual_coupling_matrix,
            channels=np.int64(self.channels),
            frequency=self.frequency,
            antennas=self.antennas,
            **interps,
        )

    def load(self, fname):
        data = np.load(fname, allow_pickle=True)
        self.mutual_coupling_matrix = data["mutual_coupling_matrix"]
        self.channels = data["channels"]
        self.parameters["frequency"] = data["frequency"]
        self.antennas = data["antennas"]
        self.interpolated = [data[f"interp{ind}"].item() for ind in range(self.channels)]

    def generate_interpolation(self, beam, ind=None, polarization=None, resolution=100):
        assert isinstance(beam, Array), "Can only interpolate arrays"
        self.channels = beam.channels
        self.mutual_coupling_matrix = beam.mutual_coupling_matrix.copy()
        self.antennas = beam.antennas.copy()

        params, shape = self.get_parameters(ind, named=True, max_vectors=0)
        self.frequency = params["frequency"]

        wavelength = scipy.constants.c / params["frequency"]
        if len(wavelength.shape) > 0:
            wavelength = wavelength[0]

        kpx = np.linspace(-2.0, 2.0, num=resolution)
        kpy = np.linspace(-2.0, 2.0, num=resolution)
        kpz = np.linspace(-2.0, 2.0, num=resolution)

        size = resolution**3

        xv, yv, zv = np.meshgrid(kpx, kpy, kpz, sparse=False, indexing="ij")
        kp = np.empty((3, size), dtype=np.float64)
        kp[0, :] = xv.reshape(1, size)
        kp[1, :] = yv.reshape(1, size)
        kp[2, :] = zv.reshape(1, size)
        norm = np.linalg.norm(kp, axis=0)
        inds = norm <= 2.0

        psi = np.zeros((size,), dtype=np.complex128)
        # r in meters, divide by lambda
        self.interpolated = [None] * self.channels
        for i in range(self.channels):
            subg_response = plane_wave_compund(kp[:, inds], beam.antennas[:, :, i] / wavelength)
            psi[inds] = subg_response.sum(axis=0).T

            self.interpolated[i] = scipy.interpolate.RegularGridInterpolator(
                (kpx, kpy, kpz),
                psi.reshape(resolution, resolution, resolution).T,
            )

    def signals(self, k, polarization, ind=None, **kwargs):
        """Complex voltage output signals after summation of antennas.

        Returns
        -------
        numpy.ndarray
            `(c,2,num_k)` ndarray where `c` is the number of channels
            requested, `2` are the two polarization axis of the Jones vector
            and `num_k` is the number of input wave vectors. If `num_k = 1`
            the returned ndarray is `(c,2)`.
        """
        k_len = k.shape[1] if len(k.shape) == 2 else 1
        assert len(k.shape) <= 2, "'k' can only be vectorized with one additional axis"
        params, shape = self.get_parameters(ind, named=True, max_vectors=0)
        if k_len == 1:
            k = k.reshape(3, 1)
        p = params["pointing"].reshape(3)

        kn = k / np.linalg.norm(k, axis=0)

        wave = np.zeros((self.channels, k_len), dtype=np.complex128)
        for i in range(self.channels):
            wave[i, :] = self.interpolated[i](
                kn[0, ...] - p[0],
                kn[1, ...] - p[1],
                kn[2, ...] - p[2],
                grid=False,
            )

        # broadcast to polarizations
        psi = wave[..., None] * polarization[None, None, :]

        if self.mutual_coupling_matrix is not None:
            psi[:, 0, ...] = self.mutual_coupling_matrix @ psi[:, 0, ...]
            psi[:, 1, ...] = self.mutual_coupling_matrix @ psi[:, 1, ...]

        ant_response = self.antenna_element(kn, polarization)

        psi[:, 0, ...] *= ant_response[None, 0, ...]
        psi[:, 1, ...] *= ant_response[None, 1, ...]

        if len(k.shape) == 1:
            psi = psi.reshape(psi.shape[:-1])

        return psi

    def gain(self, k, ind=None, polarization=None, **kwargs):
        """Gain of the antenna array."""

        if polarization is None:
            polarization = self.polarization
        elif not np.all(np.iscomplex(polarization)):
            polarization = polarization.astype(np.complex128)

        G = self.signals(k, polarization, ind=ind, **kwargs)

        lin_pol_check = np.abs(self.polarization) < 1e-6
        if not np.any(lin_pol_check):
            pol_comp = self.polarization[0] * self.polarization[1].conj()
            pol_comp = pol_comp.conj() / np.abs(pol_comp)
            G[:, 0, ...] *= pol_comp  # align polarizations

        G = np.sum(G, axis=1)  # coherent intergeneration over polarization
        G = np.sum(G, axis=0)  # coherent intergeneration over channels
        return np.abs(G)
