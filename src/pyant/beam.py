#!/usr/bin/env python

"""Defines an antenna's or entire radar system's radiation pattern"""
from abc import ABC, abstractmethod
from typing import Generic
import spacecoords.spherical as sph
from .types import NDArray_3xN, NDArray_3, NDArray_N, P, SizeError


def get_and_validate_k_shape(param_size: int | None, k: NDArray_3xN | NDArray_3) -> int | None:
    """Helper function to validate the input direction vector shape is correct.
    It returns the size of the second dimension of the k vector, if it does not
    have this dimension then it returns None.
    """
    k_len = k.shape[1] if len(k.shape) > 1 else None

    if len(k.shape) > 2 or k.shape[0] != 3:
        raise SizeError(f"k vector shape cannot be {k.shape}")

    if param_size is not None and k_len is not None and param_size != k_len:
        raise SizeError(f"parameter size {param_size} != k vector size {k.shape}")
    return k_len


class Beam(ABC, Generic[P]):
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

    def copy(self):
        """Return a copy of the current instance."""
        raise NotImplementedError("")

    @abstractmethod
    def gain(self, k: NDArray_3xN | NDArray_3, parameters: P) -> NDArray_N | float:
        """Return the gain in the given direction. This method should be
        vectorized in the `k` variable.

        Parameters
        ----------
        k
            Direction in local coordinates to evaluate
            gain in. Must be a `(3,)` vector or a `(3,n)` matrix.
        parameters
            A dataclass with the parameters needed to calculate the gain

        Returns
        -------
            Radar gain in the given direction. If input is a `(3,)`
            vector, output is a float. If input is a `(3,n)` matrix output
            is a `(n,)` vector of gains.

        """
        pass

    def sph_gain(
        self,
        azimuth: NDArray_N | float,
        elevation: NDArray_N | float,
        parameters: P,
        degrees: bool = False,
    ) -> NDArray_N | float:
        """Return the gain in the given direction.

        Parameters
        ----------
        azimuth
            Azimuth east of north to evaluate gain in.
        elevation
            Elevation from horizon to evaluate gain in.
        parameters
            A dataclass with the parameters needed to calculate the gain
        degrees
            If :code:`True` all input/output angles are in degrees,
            else they are in radians. Defaults to instance
            settings :code:`self.radians`.

        Returns
        -------
            Radar gain in the given direction. If input is a `(3,)`
            vector, output is a float. If input is a `(3,n)` matrix output
            is a `(n,)` vector of gains.
        """
        k_ang = sph.az_el_to_sph(azimuth, elevation)
        k = sph.sph_to_cart(k_ang, degrees=degrees)
        return self.gain(k, parameters)
