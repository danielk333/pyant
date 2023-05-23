#!/usr/bin/env python

'''Defines an antenna's or entire radar system's radiation pattern
'''
import numpy as np
from abc import ABC, abstractmethod
import collections


import scipy.constants

from . import coordinates


class Beam(ABC):
    '''Defines the radiation pattern, i.e gain, of a radar. Gain here means
    amplification of the electromagnetic wave amplitude when transferred to a
    complex voltage amplitude.

    :param float frequency: Frequency of radiation pattern.
    :param float azimuth: Azimuth of pointing direction.
    :param float elevation: Elevation of pointing direction.
    :param bool radians: If :code:`True` all input/output angles are in
        radians, else they are in degrees

    :ivar float frequency: Frequency of radiation pattern.
    :ivar float azimuth: Azimuth of pointing direction.
    :ivar float elevation: Elevation of pointing direction.
    :ivar bool radians: If :code:`True` all input/output angles are in
        radians, else they are in degrees
    :ivar numpy.ndarray pointing: Cartesian vector in local coordinates
        describing pointing direction.
    '''

    def __init__(self, azimuth, elevation, frequency, degrees=False, **kwargs):
        '''Basic constructor.
        '''
        self.__parameters = []
        self.__param_axis = []

        if isinstance(frequency, list) or isinstance(frequency, tuple):
            frequency = np.array(frequency, dtype=np.float64)

        sph = Beam._azel_to_numpy(azimuth, elevation)

        self.frequency = frequency
        self._azimuth = sph[0, ...]
        self._elevation = sph[1, ...]
        self.degrees = degrees

        self.pointing = coordinates.sph_to_cart(sph, degrees=degrees)

        self.register_parameter('pointing', numpy_parameter_axis=1)
        self.register_parameter('frequency')

    @property
    def azimuth(self):
        '''Azimuth, east of north, of pointing direction
        '''
        return self._azimuth

    @azimuth.setter
    def azimuth(self, val):
        sph = Beam._azel_to_numpy(val, self._elevation)
        self._azimuth = sph[0, ...]
        self.pointing = coordinates.sph_to_cart(sph, degrees = self.degrees)

    @property
    def elevation(self):
        '''Elevation from horizon of pointing direction
        '''
        return self._elevation

    @elevation.setter
    def elevation(self, val):
        sph = Beam._azel_to_numpy(self._azimuth, val)
        self._elevation = sph[1, ...]
        self.pointing = coordinates.sph_to_cart(sph, degrees = self.degrees)

    def __get_len(self, pind):
        obj = getattr(self, self.__parameters[pind])

        if isinstance(obj, np.ndarray):
            if len(obj.shape) == 0:
                return None
            elif len(obj.shape)-1 < self.__param_axis[pind]:
                return None
            else:
                return obj.shape[self.__param_axis[pind]]
        elif isinstance(obj, list) or isinstance(obj, tuple):
            return len(obj)
        else:
            return None

    def register_parameter(self, name, numpy_parameter_axis=0):
        '''Register parameter, it can then be indexed in gain calls.
        '''
        self.__parameters.append(name)
        self.__param_axis.append(numpy_parameter_axis)

    def unregister_parameter(self, name):
        '''Unregister parameter, they can no longer be indexed in gain calls.
        Can be done for speed to reduce overhead of the gain call.
        '''
        pid = self.__parameters.index(name)
        del self.__parameters[pid]
        del self.__param_axis[pid]

    def named_shape(self):
        '''Return the named shape of all variables. This can be overridden to
        extend the possible variables contained in an instance.
        '''
        kw = {self.__parameters[pind]: self.__get_len(
            pind) for pind in range(len(self.__parameters))}
        return collections.OrderedDict(**kw)

    @property
    def shape(self):
        '''The additional dimensions added to the output Gain.
        '''
        return tuple(self.__get_len(pind) for pind in range(len(self.__parameters)))

    @property
    def parameters(self):
        '''Current list of parameters.
        '''
        return tuple(self.__parameters)

    def get_parameters(
                    self, ind,
                    named=False, vectorized_parameters=False,
                    **kwargs
                ):
        '''Get parameters for a specific configuration given by `ind`.

        :param (iterable, dict, None) ind: Index for the parameters.
            Can be an iterable over the parameters, a dict with each
            parameter name and index or None for all parameters.
        :param bool named: Return parameters as a dict instead of a tuple.
        :param bool vectorized_parameters: Parameters can be vectors of
            values, used to vectorize gain calculations
            using arrays of parameters.
        :param dict **kwargs: Any keyword arugment here is used as the
            parameter here instead of the ones stored inside the Beam,
            this can be used to call gain functions programtically
            without modifying the object.

        '''
        if len(self.__parameters) == 0:
            return ()

        # only parameter that has two names, handle special case
        if 'pointing' not in kwargs:
            if 'azimuth' in kwargs or 'elevation' in kwargs:
                if not ('azimuth' in kwargs and 'elevation' in kwargs):
                    raise ValueError('If azimuth and elevation is used \
                        instead of pointing both must be supplied.')
                sph_ = Beam._azel_to_numpy(
                    kwargs['azimuth'], kwargs['elevation'])
                kwargs['pointing'] = coordinates.sph_to_cart(
                    sph_, degrees = self.degrees)
        else:
            if 'azimuth' in kwargs or 'elevation' in kwargs:
                raise ValueError('Cannot give pointing vector \
                    and angles simultaneously.')

        ind, named_shape = self.convert_ind(ind)

        params = ()
        for pind in range(len(self.__parameters)):
            key = self.__parameters[pind]

            if key in kwargs:
                if key in ind:
                    if ind[key] is not None:
                        raise ValueError('Cannot both supply keyword argument \
                            and index for parameter')
                params = params + (kwargs[key],)
                continue

            obj = getattr(self, key)

            if named_shape[key] is None:
                if key in ind:
                    if not (ind[key] is None or ind[key] == 0):
                        raise ValueError(
                            f'Cannot index scalar parameter "{key}"'
                        )
                params = params + (obj,)
            else:
                if key not in ind:
                    if named_shape[key] == 1:
                        params = params + (obj[0],)
                        continue
                    elif vectorized_parameters:
                        params = params + (obj,)
                        continue
                    else:
                        raise ValueError(
                            'Not enough parameter values or indices given'
                        )
                if isinstance(obj, np.ndarray):
                    inds = [slice(None)]*obj.ndim
                    inds[self.__param_axis[pind]] = ind[key]
                    params = params + (obj[tuple(inds)],)
                else:
                    params = params + (obj[ind[key]],)

        if named:
            kw = {self.__parameters[pind]: params[pind]
                  for pind in range(len(self.__parameters))}
            params = collections.OrderedDict(**kw)
        return params

    def convert_ind(self, ind):
        '''Convert a parameter index to a common
        {parameter namme: parameter index} dict format.
        '''
        shape = self.named_shape()
        parameters = self.parameters

        if ind is None:
            ind = {}
        elif isinstance(ind, tuple) or isinstance(ind, list):
            if len(ind) != len(parameters):
                raise ValueError(f'Not enough incidences ({len(ind)}) \
                    given to choose ({len(parameters)}) parameters')
            ind = {key: i for key, i in zip(parameters, ind)}
        elif isinstance(ind, dict):
            pass
        else:
            raise ValueError(f'Indexing of type "{type(ind)}" not supported')

        return ind, shape

    def _check_degrees(self, azimuth, elevation, degrees):
        '''Converts input azimuth and elevation to the correct angle units.
        '''
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
        '''Convert input azimuth and elevation iterables to
        spherical coordinates states, i.e a `shape=(3,n)` numpy array.
        '''

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

    @property
    def wavelength(self):
        '''The radar wavelength.
        '''
        return scipy.constants.c/self.frequency

    @wavelength.setter
    def wavelength(self, val):
        self.frequency = scipy.constants.c/val

    def copy(self):
        '''Return a copy of the current instance.
        '''
        raise NotImplementedError('')

    def complex(self, k, ind=None, polarization=None, **kwargs):
        '''The complex voltage output can be implemented as a middle step in
        gain calculation. Can include polarization channels.
        '''
        raise NotImplementedError('')

    def sph_point(self, azimuth, elevation, degrees=None):
        '''Point beam towards azimuth and elevation coordinate.

        :param float azimuth: Azimuth east of north of pointing direction.
        :param float elevation: Elevation from horizon of pointing direction.
        :param bool radians: If :code:`True` all input/output angles are in
            radians, if False degrees are used. Defaults to instance
            settings :code:`self.radians`.
        :return: :code:`None`

        '''
        azimuth, elevation = self._check_degrees(azimuth, elevation, degrees)
        sph = Beam._azel_to_numpy(azimuth, elevation)

        self._azimuth = azimuth
        self._elevation = elevation
        self.pointing = coordinates.sph_to_cart(sph, degrees = self.degrees)

    def point(self, k):
        '''Point beam in local cartesian direction.

        :param numpy.ndarray k: Pointing direction in local coordinates.
        :return: :code:`None`
        '''
        self.pointing = k/np.linalg.norm(k, axis=0)
        sph = coordinates.cart_to_sph(
            self.pointing,
            degrees = self.degrees,
        )
        self._azimuth = sph[0, ...]
        self._elevation = sph[1, ...]

    def sph_angle(self, azimuth, elevation, degrees=None):
        '''Get angle between azimuth and elevation and pointing direction.

        :param float azimuth: Azimuth east of north to measure from.
        :param float elevation: Elevation from horizon to measure from.
        :param bool radians: If :code:`True` all input/output angles are in
            radians, if False degrees are used. Defaults to instance
            settings :code:`self.radians`.

        :return: Angle between pointing and given direction.
        :rtype: float
        '''
        if degrees is None:
            degrees = self.degrees

        sph = Beam._azel_to_numpy(azimuth, elevation)
        k = coordinates.sph_to_cart(sph, degrees=degrees)
        return self.angle(k, degrees=degrees)

    def angle(self, k, degrees=None):
        '''Get angle between local direction and pointing direction.

        :param numpy.ndarray k: Direction to evaluate angle to.
        :param bool radians: If :code:`True` all input/output angles are in
            radians, if False degrees are used. Defaults to instance
            settings :code:`self.radians`.

        :return: Angle between pointing and given direction.
        :rtype: float
        '''
        if degrees is None:
            degrees = self.degrees

        if len(self.pointing.shape) > 1:
            if len(k.shape) > 1:
                theta = np.empty(
                    (k.shape[1], self.pointing.shape[1]), dtype=k.dtype)
            else:
                theta = np.empty((self.pointing.shape[1], ), dtype=k.dtype)

            for ind in range(self.pointing.shape[1]):
                theta[..., ind] = coordinates.vector_angle(
                    self.pointing[ind, :], k, degrees=degrees)
        else:
            theta = coordinates.vector_angle(self.pointing, k, degrees=degrees)

        return theta

    @abstractmethod
    def gain(
                    self, k,
                    ind=None, polarization=None,
                    vectorized_parameters=False,
                    **kwargs
                ):
        '''Return the gain in the given direction. This method should be
        vectorized in the `k` variable.

        If e.g. pointing is the only parameter with 5 directions,
        :code:`ind=(2,)` would evaluate the gain using the third
        pointing direction.

        :param numpy.ndarray k: Direction in local coordinates to evaluate
            gain in. Must be a `(3,)` vector or a `(3,n)` matrix.
        :param tuple ind: The incidences of the available parameters.
            If the parameters have a size of `1`, no index is needed.
        :param numpy.ndarray polarization: The Jones vector of the incoming
            plane waves, if applicable for the beam in question.

        :return: Radar gain in the given direction. If input is a `(3,)`
            vector, output is a float. If input is a `(3,n)` matrix output
            is a `(n,)` vector of gains.
        :rtype: float/numpy.ndarray
        '''
        pass

    def sph_gain(
                    self, azimuth, elevation,
                    ind=None, polarization=None,
                    degrees=None, vectorized_parameters=False,
                    **kwargs
                ):
        '''Return the gain in the given direction.

        :param float azimuth: Azimuth east of north to evaluate gain in.
        :param float elevation: Elevation from horizon to evaluate gain in.
        :param bool radians: If :code:`True` all input/output angles are in
            radians, if False degrees are used. Defaults to instance
            settings :code:`self.radians`.

        :return: Radar gain in the given direction.
        :rtype: float
        '''
        if degrees is None:
            degrees = self.degrees

        sph = Beam._azel_to_numpy(azimuth, elevation)

        k = coordinates.sph_to_cart(sph, degrees=degrees)
        return self.gain(
            k,
            polarization=polarization,
            ind=ind,
            vectorized_parameters=vectorized_parameters,
            **kwargs
        )
