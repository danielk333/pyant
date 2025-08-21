#!/usr/bin/env python

"""Defines an antenna's or entire radar system's radiation pattern"""
from typing import Any
import functools
import operator
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
    Parameters
        #todo

    """

    def __init__(self, parameters: dict[str, NDArray | float] | None = None):
        """Basic constructor."""
        self.parameters = collections.OrderedDict()
        if parameters is not None:
            self.parameters.update(parameters)
        self.meta: dict[str, Any] = {}

    @property
    def frequency(self):
        """The radar wavelength."""
        assert "frequency" in self.parameters
        return self.parameters["frequency"]

    @frequency.setter
    def frequency(self, val: NDArray | float):
        self.parameters["frequency"] = val

    @property
    def wavelength(self):
        """The radar wavelength."""
        return scipy.constants.c / self.frequency

    @wavelength.setter
    def wavelength(self, val: NDArray | float):
        self.frequency = scipy.constants.c / val

    def _get_parameter_len(self, key: str):
        """Get the length of a parameter axis"""
        obj = self.parameters[key]
        if isinstance(obj, float):
            return 0
        else:
            return obj.shape[0]

    @property
    def shape(self):
        """The additional dimensions added to the output Gain if broadcasting is enabled."""
        return tuple(self._get_parameter_len(key) for key in self.parameters)

    @property
    def keys(self):
        """Current list of parameters."""
        return self.parameters.keys()

    @staticmethod
    def _azel_to_numpy(azimuth: NDArray | float, elevation: NDArray | float) -> NDArray:
        """Convert input azimuth and elevation to spherical coordinates states,
        i.e a `shape=(3,n)` numpy array.
        """

        if isinstance(azimuth, float):
            az_len = None
        else:
            az_len = azimuth.size

        if isinstance(elevation, float):
            el_len = None
        else:
            el_len = elevation.size

        if el_len is not None and az_len is not None:
            assert el_len == az_len, f"azimuth {az_len} and elevation {el_len} sizes must agree"

        if az_len is not None:
            shape = (3, az_len)
        elif el_len is not None:
            shape = (3, el_len)
        else:
            shape = (3, 1)

        sph = np.empty(shape, dtype=np.float64)
        sph[0, ...] = azimuth
        sph[1, ...] = elevation
        sph[2, ...] = 1.0

        return sph

    def copy(self):
        """Return a copy of the current instance."""
        raise NotImplementedError("")

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
        """Point beam in local cartesian direction.

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
        if degrees is None:
            degrees = self.degrees

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
        if degrees is None:
            degrees = self.degrees

        pt: NDArray = self.parameters["pointing"]

        if len(pt.shape) > 1:
            if len(k.shape) > 1:
                theta = np.empty((k.shape[1], pt.shape[1]), dtype=k.dtype)
            else:
                theta = np.empty((pt.shape[1],), dtype=k.dtype)

            for ind in range(pt.shape[1]):
                theta[..., ind] = coordinates.vector_angle(pt[ind, :], k, degrees=degrees)
        else:
            theta = coordinates.vector_angle(pt, k, degrees=degrees)

        return theta

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
        if degrees is None:
            degrees = self.degrees

        sph = Beam._azel_to_numpy(azimuth, elevation)

        k = coordinates.sph_to_cart(sph, degrees=degrees)
        return self.gain(
            k,
            polarization=polarization,
        )
