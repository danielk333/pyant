#!/usr/bin/env python

"""Defines an antenna's or entire radar system's radiation pattern"""
from abc import ABC, abstractmethod
import collections

import numpy as np
from numpy.typing import NDArray
import scipy.constants

from . import coordinates


class Beam(ABC):
    """Defines the radiation pattern, i.e gain, of a radar. Gain here means
    amplification of the electromagnetic wave amplitude when transferred to a
    complex voltage amplitude.

    Notes
    ------
    There are four possible ways of broadcasting input arrays over the parameters:

    The additional axis of all parameters lines up with the input wave vectors additional
    axis size, in which case each input k-vector gets evaluated versus each set of
    parameters.
    Input size (3, n), parameter shapes (..., n), output size (n,).

    The parameters are all scalars and the input k-vector gets evaluated over this single set.
    Input size (3, n), parameter shapes (...), output size (n,).

    The additional axis of all parameters line up and the input k-vector is a single vector,
    in which case this vector gets computed for all sets of parameters.
    Input size (3,), parameter shapes (..., n), output size (n,).

    The parameters are all scalars and the input k-vector is a single vector.
    Input size (3,), parameter shapes (...), output size ().

    These ways allow for any set of computations and broadcasts (although they need to be prepared
    outside the scope of this class) to be set-up using the Beam interface.

    """

    def __init__(self):
        """Basic constructor."""
        self.parameters = collections.OrderedDict()
        self.parameters_shape = {}

    def _get_parameter_len(self, key: str):
        """Get the length of a parameter axis, its always the last array dimension"""
        obj = self.parameters[key]
        if isinstance(obj, np.ndarray):
            if key in self.parameters_shape:
                shape = self.parameters_shape[key]
                if len(obj.shape) == len(shape):
                    return 0
                else:
                    return obj.shape[-1]
            else:
                return obj.shape[-1]
        else:
            return 0

    @property
    def size(self):
        """The additional dimensions added to the output Gain if broadcasting is enabled."""
        shape = [self._get_parameter_len(key) for key in self.parameters]
        if len(shape) == 0:
            return 0
        assert all(x == shape[0] for x in shape), (
            "all parameter shapes must line up:"
            f"{list(self.parameters.keys())} -> {shape}"
        )
        return shape[0]

    def validate_parameter_shapes(self):
        """Helper function to validate the input parameter shapes are correct"""
        # TODO: maybe change to raise custom exceptions?
        size = None
        for key, p in self.parameters.items():
            if size is None:
                size = self._get_parameter_len(key)
            assert size == self._get_parameter_len(key), "all parameter shapes must line up"
            if key in self.parameters_shape:
                shape = self.parameters_shape[key]
                assert len(p.shape) <= len(shape) + 1 and len(p.shape) >= len(
                    shape
                ), f"{key} can only have {len(shape)} or {len(shape) + 1} axis, not {len(p.shape)}"
                assert (
                    p.shape[: len(shape)] == shape
                ), f"{key} needs at least {shape} dimensions, not {p.shape}"

    def validate_k_shape(self, k):
        """Helper function to validate the input direction vector shape is correct"""
        # TODO: maybe change to raise custom exceptions?
        size = self.size
        k_len = k.shape[1] if len(k.shape) > 1 else 0
        if size > 0:
            assert (
                size == k_len or k_len == 0
            ), "input k vector must either be single vector or line up with parameter dimensions"
        assert len(k.shape) <= 2, "k vector can only be vectorized along one extra axis"
        assert k.shape[0] == 3, f"pointing vector must at least be a 3-vector, not {k.shape[0]}"
        return k_len

    @property
    def keys(self):
        """Current list of parameters."""
        return self.parameters.keys()

    @staticmethod
    def _azel_to_numpy(azimuth: NDArray | float, elevation: NDArray | float) -> NDArray:
        """Convert input azimuth and elevation to spherical coordinates states,
        i.e a `shape=(3,n)` numpy array.
        """

        az_len = azimuth.size if isinstance(azimuth, np.ndarray) else None
        el_len = elevation.size if isinstance(elevation, np.ndarray) else None

        if el_len is not None and az_len is not None:
            assert el_len == az_len, f"azimuth {az_len} and elevation {el_len} sizes must agree"

        if az_len is not None:
            shape = (3, az_len)
        elif el_len is not None:
            shape = (3, el_len)
        else:
            shape = (3, )

        sph = np.empty(shape, dtype=np.float64)
        sph[0, ...] = azimuth
        sph[1, ...] = elevation
        sph[2, ...] = 1.0

        return sph

    def copy(self):
        """Return a copy of the current instance."""
        raise NotImplementedError("")

    def to_h5(self, path):
        """Write defining parameters to a h5 file"""
        raise NotImplementedError("")

    @classmethod
    def from_h5(cls):
        """Load defining parameters from a h5 file and instantiate a beam"""
        raise NotImplementedError("")

    @property
    def frequency(self):
        """The radar wavelength."""
        return self.parameters["frequency"]

    @frequency.setter
    def frequency(self, val):
        self.parameters["frequency"] = val

    @property
    def wavelength(self):
        """The radar wavelength."""
        return scipy.constants.c / self.frequency

    @wavelength.setter
    def wavelength(self, val):
        self.frequency = scipy.constants.c / val

    def sph_point(
        self, azimuth: NDArray | float, elevation: NDArray | float, degrees: bool = False
    ):
        """Point beam towards azimuth and elevation coordinate.

        Parameters
        ----------
        azimuth : float
            Azimuth east of north of pointing direction.
        elevation : float
            Elevation from horizon of pointing direction.
        degrees : bool
            If :code:`True` all input/output angles are in degrees,
            else they are in radians. Defaults to instance
            settings :code:`self.radians`.

        """
        sph = Beam._azel_to_numpy(azimuth, elevation)
        self.parameters["pointing"] = coordinates.sph_to_cart(sph, degrees=degrees)

    def point(self, k: NDArray):
        """Point beam in local Cartesian direction.

        Parameters
        ----------
        k : numpy.ndarray
            Pointing direction in local coordinates.

        """
        self.parameters["pointing"] = k / np.linalg.norm(k, axis=0)

    def sph_angle(
        self, azimuth: NDArray | float, elevation: NDArray | float, degrees: bool = False
    ) -> NDArray | float:
        """Get angle between azimuth and elevation and pointing direction.

        Parameters
        ----------
        azimuth : float or NDArray
            Azimuth east of north of pointing direction.
        elevation : float or NDArray
            Elevation from horizon of pointing direction.
        degrees : bool
            If :code:`True` all input/output angles are in degrees,
            else they are in radians.

        Returns
        -------
        float or NDArray
            Angle between pointing and given direction.

        """
        sph = Beam._azel_to_numpy(azimuth, elevation)
        k = coordinates.sph_to_cart(sph, degrees=degrees)
        return self.angle(k, degrees=degrees)

    def angle(self, k: NDArray, degrees: bool = False) -> NDArray | float:
        """Get angle between local direction and pointing direction.

        Parameters
        ----------
        k : numpy.ndarray
            Direction to evaluate angle to.
        degrees : bool
            If :code:`True` all input/output angles are in degrees,
            else they are in radians. Defaults to instance
            settings :code:`self.radians`.

        Returns
        -------
        float or NDArray
            Angle between pointing and given direction.

        """
        pt: NDArray = self.parameters["pointing"]
        return coordinates.vector_angle(pt, k, degrees=degrees)

    @abstractmethod
    def gain(self, k: NDArray, polarization: NDArray | None = None):
        """Return the gain in the given direction. This method should be
        vectorized in the `k` variable.

        Parameters
        ----------
        k : numpy.ndarray
            Direction in local coordinates to evaluate
            gain in. Must be a `(3,)` vector or a `(3,n)` matrix.
        polarization : numpy.ndarray
            The Jones vector of the incoming
            plane waves, if applicable for the beam in question.

        Returns
        -------
        float/numpy.ndarray
            Radar gain in the given direction. If input is a `(3,)`
            vector, output is a float. If input is a `(3,n)` matrix output
            is a `(n,)` vector of gains.

        """
        pass

    def sph_gain(
        self,
        azimuth: NDArray | float,
        elevation: NDArray | float,
        polarization: NDArray | None = None,
        degrees: bool = False,
    ):
        """Return the gain in the given direction.

        Parameters
        ----------
        azimuth : float
            Azimuth east of north to evaluate gain in.
        elevation : float
            Elevation from horizon to evaluate gain in.
        degrees : bool
            If :code:`True` all input/output angles are in degrees,
            else they are in radians. Defaults to instance
            settings :code:`self.radians`.

        Returns
        -------
        float/numpy.ndarray
            Radar gain in the given direction. If input is a `(3,)`
            vector, output is a float. If input is a `(3,n)` matrix output
            is a `(n,)` vector of gains.
        """
        sph = Beam._azel_to_numpy(azimuth, elevation)

        k = coordinates.sph_to_cart(sph, degrees=degrees)
        return self.gain(
            k,
            polarization=polarization,
        )
