#!/usr/bin/env python

"""Defines an antenna's or entire radar system's radiation pattern
"""
import functools
import operator
from abc import ABC, abstractmethod
import collections

import numpy as np
import scipy.constants

from . import coordinates


class Beam(ABC):
    """Defines the radiation pattern, i.e gain, of a radar. Gain here means
    amplification of the electromagnetic wave amplitude when transferred to a
    complex voltage amplitude.

    Parameters
    ----------
    frequency : float
        Frequency of radiation pattern.
    azimuth : float
        Azimuth of pointing direction.
    elevation : float
        Elevation of pointing direction.
    degrees : bool
        If :code:`True` all input/output angles are in degrees,
        else they are in radians.

    Attributes
    ----------
    frequency : float
        Frequency of radiation pattern.
    azimuth : float
        Azimuth of pointing direction.
    elevation : float
        Elevation of pointing direction.
    degrees : bool
        If :code:`True` all input/output angles are in degrees,
        else they are in radians.
    pointing : numpy.ndarray
        Cartesian vector in local coordinates describing pointing direction.

    Notes
    ------
    Parameters
        Parameters are stored and indexed as coordinate axis similar to how NetCDF
        enforces the definition of matrix axis. By default, if the axis consist of
        multiple points a matrix of gains will be generated for each evaluation of
        the gain. If there are multiple input wave vectors into the gain evaluation
        till will be appended at the end of the array as a new axis dimension.
        One can also skip the evaluation of the entire matrix by supplying slices
        of the parameter axis.

    """

    def __init__(self, azimuth, elevation, frequency, degrees=False, **kwargs):
        """Basic constructor."""
        self._keys = []
        self._inds = {}
        self._parameter_axis = {}
        self.parameters = collections.OrderedDict()
        self.degrees = degrees

        self.register_parameter("pointing", numpy_parameter_axis=1)
        self.register_parameter("frequency")

        self.frequency = frequency
        self.sph_point(azimuth, elevation)

    @property
    def pointing(self):
        return self.parameters["pointing"]

    @pointing.setter
    def pointing(self, val):
        if len(val.shape) <= 1:
            val = val.reshape(3, 1)
        sph = coordinates.cart_to_sph(val, degrees=self.degrees)
        self.fill_parameter("pointing", val)
        self._azimuth = sph[0, ...]
        self._elevation = sph[1, ...]

    @property
    def azimuth(self):
        """Azimuth, east of north, of pointing direction"""
        return self._azimuth

    @azimuth.setter
    def azimuth(self, val):
        sph = Beam._azel_to_numpy(val, self._elevation)
        if len(sph.shape) <= 1:
            sph = sph.reshape(3, 1)
        self._azimuth = sph[0, ...]
        pointing = coordinates.sph_to_cart(sph, degrees=self.degrees)
        self.fill_parameter("pointing", pointing)

    @property
    def elevation(self):
        """Elevation from horizon of pointing direction"""
        return self._elevation

    @elevation.setter
    def elevation(self, val):
        sph = Beam._azel_to_numpy(self._azimuth, val)
        if len(sph.shape) <= 1:
            sph = sph.reshape(3, 1)
        self._elevation = sph[1, ...]
        pointing = coordinates.sph_to_cart(sph, degrees=self.degrees)
        self.fill_parameter("pointing", pointing)

    @property
    def frequency(self):
        """The radar wavelength."""
        return self.parameters["frequency"]

    @frequency.setter
    def frequency(self, val):
        self.fill_parameter("frequency", val)

    @property
    def wavelength(self):
        """The radar wavelength."""
        return scipy.constants.c / self.frequency

    @wavelength.setter
    def wavelength(self, val):
        self.frequency = scipy.constants.c / val

    def fill_parameter(self, key, data):
        """Assign axis values to parameter"""
        if isinstance(data, list) or isinstance(data, tuple):
            data = np.array(data, dtype=np.float64)
        if isinstance(data, np.ndarray):
            self.parameters[key] = data
        elif isinstance(self.parameters[key], np.ndarray):
            self.parameters[key][:] = data
        else:
            self.parameters[key] = data

    def _get_parameter_len(self, key):
        """Get the length of a parameter axis, `None` indicates scalar axis."""
        obj = self.parameters[key]
        if len(obj.shape) == 0:
            return 1
        elif len(obj.shape) - 1 < self._parameter_axis[key]:
            return 1
        else:
            return obj.shape[self._parameter_axis[key]]

    def register_parameter(self, name, numpy_parameter_axis=0):
        """Register parameter, it can then be indexed in gain calls."""
        self._keys.append(name)
        self._parameter_axis[name] = numpy_parameter_axis
        self.parameters[name] = np.full((1,), np.nan, dtype=np.float64)
        self._inds[name] = len(self._keys) - 1

    def unregister_parameter(self, name):
        """Unregister parameter, they can no longer be indexed in gain calls.
        Can be done for speed to reduce overhead of the gain call.
        """
        pid = self._inds[name]
        del self._keys[pid]
        del self._parameter_axis[name]
        del self.parameters[name]
        del self._inds[name]

    @property
    def named_shape(self):
        """Return the named shape of all variables. This can be overridden to
        extend the possible variables contained in an instance.
        """
        return collections.OrderedDict([(key, self._get_parameter_len(key)) for key in self._keys])

    @property
    def shape(self):
        """The additional dimensions added to the output Gain."""
        return tuple(self._get_parameter_len(key) for key in self._keys)

    @property
    def keys(self):
        """Current list of parameters."""
        return tuple(self._keys)

    def get_parameters(self, ind=None, named=False, max_vectors=None):
        """Get parameters for a specific configuration given by `ind`.

        Parameters
        ----------
        ind : (iterable, dict, None)
            Index for the parameters.
            Can be an iterable over the parameters, a dict with each
            parameter name and index or None for all parameters.
        named : bool
            Return parameters as a dict instead of a tuple.
        max_vectors : (int, None)
            Maximum number of parameters that are allowed to be vectors.

        """
        if len(self._keys) == 0:
            if named:
                return {}
            else:
                return ()

        inds = self.ind_to_dict(ind)
        if named:
            params = collections.OrderedDict(
                {key: self.parameters[key][inds[key]].copy() for key in self._keys}
            )
            shape = {
                key: x.shape[self._parameter_axis[key]]
                if len(x.shape) > self._parameter_axis[key]
                else 0
                for key, x in params.items()
            }
            vector_cnt = sum(1 for x in shape.values() if x > 1)
        else:
            params = list(self.parameters[key][inds[key]].copy() for key in self._keys)
            shape = (
                x.shape[self._parameter_axis[key]]
                if len(x.shape) > self._parameter_axis[key]
                else 0
                for key, x in zip(self._keys, params)
            )
            vector_cnt = sum(1 for x in shape if x > 1)

        if max_vectors is not None:
            assert vector_cnt <= max_vectors, "Too many vector valued parameters"

        return params, shape

    def broadcast_params(self, params, shape, input_k_len):
        """Broadcast the input parameters to the output shape taking the input wave
        vectorization length into account.

        Notes
        -----
        Input params and shape
            This function assumes params and shape was generated using the `named=True` option.
        """
        vector_cnt = sum(1 for x in shape.values() if x > 1)
        max_shape = max(shape.values())
        assert vector_cnt <= 1, "Too many vector valued parameters to broadcast"

        gain_shape = tuple(x for x in (input_k_len, max_shape) if x > 0)
        g_max = len(gain_shape)
        if g_max == 0:
            gain_array = np.float64(np.nan)
        else:
            gain_array = np.full(gain_shape, np.nan, dtype=np.float64)
        expanded_params = {}
        for key in params:
            if self._parameter_axis[key] > 0:
                expanded_params[key] = params[key].copy()
                continue

            if g_max == 0:
                ext_param = params[key]
            elif shape[key] <= 1:
                ext_param = np.full_like(gain_array, params[key])
            else:
                ext_param = np.broadcast_to(params[key], gain_array.shape).copy()
            expanded_params[key] = ext_param

        return expanded_params, gain_array

    def ind_to_dict(self, ind):
        """Convert a parameter index to a common
        {parameter name: parameter indexing} dict format.
        """
        base_inds = {key: [slice(None)] * self.parameters[key].ndim for key in self._keys}
        if ind is None:
            pass
        elif isinstance(ind, dict):
            for key in self._keys:
                if key not in ind:
                    continue
                base_inds[key][self._parameter_axis[key]] = ind[key]
        else:
            if len(ind) != len(self._keys):
                raise ValueError(
                    f"Not enough incidences ({len(ind)}) \
                    given to choose ({len(self._keys)}) parameters"
                )
            for key, indexing in zip(self._keys, ind):
                base_inds[key][self._parameter_axis[key]] = indexing
        base_inds = {key: tuple(val) for key, val in base_inds.items()}
        return base_inds

    def _check_degrees(self, azimuth, elevation, degrees):
        """Converts input azimuth and elevation to the correct angle units."""
        if degrees is None:
            return azimuth, elevation
        else:
            if degrees == self.degrees:
                return azimuth, elevation
            else:
                if degrees:
                    return np.radians(azimuth), np.radians(elevation)
                else:
                    return np.degrees(azimuth), np.degrees(elevation)

    @staticmethod
    def _azel_to_numpy(azimuth, elevation):
        """Convert input azimuth and elevation iterables to
        spherical coordinates states, i.e a `shape=(3,n)` numpy array.
        """

        if isinstance(azimuth, np.ndarray):
            if len(azimuth.shape) == 0:
                az_len = None
            else:
                az_len = azimuth.size
        elif isinstance(azimuth, list) or isinstance(azimuth, tuple):
            az_len = len(azimuth)
        else:
            az_len = None

        if isinstance(elevation, np.ndarray):
            if len(elevation.shape) == 0:
                el_len = None
            else:
                el_len = elevation.size
        elif isinstance(elevation, list) or isinstance(elevation, tuple):
            el_len = len(elevation)
        else:
            el_len = None

        if az_len is not None:
            shape = (3, az_len)
        elif el_len is not None:
            shape = (3, el_len)
        else:
            shape = (3,)

        sph = np.empty(shape, dtype=np.float64)
        sph[0, ...] = azimuth
        sph[1, ...] = elevation
        sph[2, ...] = 1.0

        return sph

    def copy(self):
        """Return a copy of the current instance."""
        raise NotImplementedError("")

    def signals(self, k, ind=None, polarization=None, **kwargs):
        """The complex voltage output can be implemented as a middle step in
        gain calculation. Can include polarization channels.
        """
        raise NotImplementedError("")

    def sph_point(self, azimuth, elevation, degrees=None):
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
        azimuth, elevation = self._check_degrees(azimuth, elevation, degrees)
        sph = Beam._azel_to_numpy(azimuth, elevation)
        if len(sph.shape) <= 1:
            sph = sph.reshape(3, 1)
        self._azimuth = sph[0, ...]
        self._elevation = sph[1, ...]
        pointing = coordinates.sph_to_cart(sph, degrees=self.degrees)
        self.fill_parameter("pointing", pointing)

    def point(self, k):
        """Point beam in local cartesian direction.

        Parameters
        ----------
        k : numpy.ndarray
            Pointing direction in local coordinates.

        """
        self.pointing = k / np.linalg.norm(k, axis=0)

    def sph_angle(self, azimuth, elevation, degrees=None):
        """Get angle between azimuth and elevation and pointing direction.

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

        Returns
        -------
        float
            Angle between pointing and given direction.

        """
        if degrees is None:
            degrees = self.degrees

        sph = Beam._azel_to_numpy(azimuth, elevation)
        k = coordinates.sph_to_cart(sph, degrees=degrees)
        return self.angle(k, degrees=degrees)

    def angle(self, k, degrees=None):
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
        float
            Angle between pointing and given direction.

        """
        if degrees is None:
            degrees = self.degrees

        pt = self.pointing

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
    def gain(self, k, ind=None, polarization=None, **kwargs):
        """Return the gain in the given direction. This method should be
        vectorized in the `k` variable.

        Parameters
        ----------
        k : numpy.ndarray
            Direction in local coordinates to evaluate
            gain in. Must be a `(3,)` vector or a `(3,n)` matrix.
        ind : tuple
            The incidences of the available parameters.
            If the parameters have a size of `1`, no index is needed.
        polarization : numpy.ndarray
            The Jones vector of the incoming
            plane waves, if applicable for the beam in question.

        Returns
        -------
        float/numpy.ndarray
            Radar gain in the given direction. If input is a `(3,)`
            vector, output is a float. If input is a `(3,n)` matrix output
            is a `(n,)` vector of gains.

        Notes
        -----
        Parameter indexing
            If e.g. pointing is the only parameter with 5 directions,
            :code:`ind=(2,)` would evaluate the gain using the third
            pointing direction.

        """
        pass

    def sph_gain(self, azimuth, elevation, ind=None, polarization=None, degrees=None, **kwargs):
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
            ind=ind,
            **kwargs,
        )


class SummedBeams(Beam):
    def __init__(self, beams, azimuth=0, elevation=np.pi / 2, frequency=0, degrees=False):
        super().__init__(azimuth, elevation, frequency, degrees=False)
        self.beams = beams

    def gain(self, k, ind=None, polarization=None, **kwargs):
        gains = [b.gain(k, ind=ind, polarization=polarization, **kwargs) for b in self.beams]
        return functools.reduce(operator.add, gains)
