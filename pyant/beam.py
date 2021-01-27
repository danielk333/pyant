#!/usr/bin/env python

'''Defines an antenna's or entire radar system's radiation pattern

(c) 2020 Daniel Kastinen
'''
import numpy as np
from abc import ABC, abstractmethod
import collections


import scipy.constants

from . import coordinates

class Beam(ABC):
    '''Defines the radiation pattern, i.e gain, of a radar. Gain here means amplification of the electromagnetic wave amplitude when transfered to a complex voltage amplitude.

    :param float frequency: Frequency of radiation pattern.
    :param float azimuth: Azimuth of pointing direction.
    :param float elevation: Elevation of pointing direction.
    :param bool radians: If :code:`True` all input/output angles are in radians, else they are in degrees

    :ivar float frequency: Frequency of radiation pattern.
    :ivar float azimuth: Azimuth of pointing direction.
    :ivar float elevation: Elevation of pointing direction.
    :ivar bool radians: If :code:`True` all input/output angles are in radians, else they are in degrees
    :ivar numpy.ndarray pointing: Cartesian vector in local coordinates describing pointing direction.
    '''


    def __init__(self, azimuth, elevation, frequency, radians=False, **kwargs):
        '''Basic constructor.
        '''
        self.__parameters = []
        self.__param_axis = []

        if isinstance(frequency, list) or isinstance(frequency, tuple):
            frequency = np.array(frequency, dtype=np.float64)

        sph = Beam._azel_to_numpy(azimuth, elevation)

        self.frequency = frequency
        self._azimuth = sph[0,...]
        self._elevation = sph[1,...]
        self.radians = radians

        self.pointing = coordinates.sph_to_cart(sph, radians = radians)

        self.register_parameter('pointing', numpy_parameter_axis=1)
        self.register_parameter('frequency')


    @property
    def azimuth(self):
        return self._azimuth

    @azimuth.setter
    def azimuth(self, val):
        sph = Beam._azel_to_numpy(val, self._elevation)
        self._azimuth = sph[0,...]
        self.pointing = coordinates.sph_to_cart(sph, radians = self.radians)

    @property
    def elevation(self):
        return self._elevation

    @elevation.setter
    def elevation(self, val):
        sph = Beam._azel_to_numpy(self._azimuth, val)
        self._elevation = sph[1,...]
        self.pointing = coordinates.sph_to_cart(sph, radians = self.radians)


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
        '''Unregister parameter, they can no longer be indexed in gain calls. Can be done for speed to reduce overhead of the gain call.
        '''
        pid = self.__parameters.index(name)
        del self.__parameters[pid]
        del self.__param_axis[pid]


    def named_shape(self):
        '''Return the named shape of all variables. This can be overridden to extend the possible variables contained in an instance.
        '''
        kw = {self.__parameters[pind]: self.__get_len(pind) for pind in range(len(self.__parameters))}
        return collections.OrderedDict(**kw)


    @property
    def shape(self):
        '''The additional dimensions added to the output Gain.
        '''
        return tuple(self.__get_len(pind) for pind in range(len(self.__parameters)))


    @property
    def parameters(self):
        return tuple(self.__parameters)


    def get_parameters(self, ind, named=False, **kwargs):
        if len(self.__parameters) == 0:
            return ()

        #only parameter that has two names, handle special case
        if 'pointing' not in kwargs:
            if 'azimuth' in kwargs or 'elevation' in kwargs:
                if not ('azimuth' in kwargs and 'elevation' in kwargs):
                    raise ValueError('If azimuth and elevation is used instead of pointing both must be supplied.')
                sph_ = Beam._azel_to_numpy(kwargs['azimuth'], kwargs['elevation'])
                kwargs['pointing'] = coordinates.sph_to_cart(sph_, radians = self.radians)
        else:
            if 'azimuth' in kwargs or 'elevation' in kwargs:
                raise ValueError('Cannot give pointing vector and angles simultaneously.')

        ind, named_shape = self.convert_ind(ind)

        params = ()
        for pind in range(len(self.__parameters)):
            key = self.__parameters[pind]

            if key in kwargs:
                if key in ind:
                    if ind[key] is not None:
                        raise ValueError('Cannot both supply keyword argument and index for parameter')
                params += (kwargs[key],)
                continue

            obj = getattr(self, key)

            if named_shape[key] is None:
                if key in ind:
                    if not (ind[key] is None or ind[key] == 0):
                        raise ValueError(f'Cannot index scalar parameter "{key}"')
                params += (obj,)
            else:
                if key not in ind:
                    if named_shape[key] == 1:
                        params += (obj[0],)
                        continue
                    else:
                        raise ValueError(f'Not enough parameter values or indices given')
                if isinstance(obj, np.ndarray):
                    I = [slice(None)]*obj.ndim
                    I[self.__param_axis[pind]] = ind[key]
                    params += (obj[tuple(I)],)
                else:
                    params += (obj[ind[key]],)

        if named:
            kw = {self.__parameters[pind]: params[pind] for pind in range(len(self.__parameters))}
            params = collections.OrderedDict(**kw)
        return params


    def convert_ind(self, ind):
        shape = self.named_shape()
        parameters = self.parameters

        if ind is None:
            ind = {}
        elif isinstance(ind, tuple) or isinstance(ind, list):
            if len(ind) != len(parameters):
                raise ValueError(f'Not enough incidences ({len(ind)}) given to choose ({len(parameters)}) parameters')
            ind = {key:i for key,i in zip(parameters, ind)}
        elif isinstance(ind, dict):
            pass
        else:
            raise ValueError(f'Indexing of type "{type(ind)}" not supported')

        return ind, shape


    def _check_radians(self, azimuth, elevation, radians):
        if radians is None:
            return azimuth, elevation
        else:
            if radians == self.radians:
                return azimuth, elevation
            else:
                if radians:
                    return np.degrees(azimuth), np.degrees(elevation)
                else:
                    return np.radians(azimuth), np.radians(elevation)

    @staticmethod
    def _azel_to_numpy(azimuth, elevation):

        if isinstance(azimuth, np.ndarray):
            if len(azimuth.shape) == 0:
                az_len = None
            else:
                return azimuth.size
        elif isinstance(azimuth, list) or isinstance(azimuth, tuple):
            az_len = len(azimuth)
        else:
            az_len = None

        if isinstance(elevation, np.ndarray):
            if len(elevation.shape) == 0:
                el_len = None
            else:
                return elevation.size
        elif isinstance(elevation, list) or isinstance(elevation, tuple):
            el_len = len(elevation)
        else:
            el_len = None

        if az_len is not None:
            shape = (3,az_len)
        elif el_len is not None:
            shape = (3,el_len)
        else:
            shape = (3,)

        sph = np.empty(shape, dtype=np.float64)
        sph[0,...] = azimuth
        sph[1,...] = elevation
        sph[2,...] = 1.0

        return sph


    @property
    def wavelength(self):
        return scipy.constants.c/self.frequency


    @wavelength.setter
    def wavelength(self, val):
        self.frequency = scipy.constants.c/val


    def copy(self):
        '''Return a copy of the current instance.
        '''
        raise NotImplementedError('')


    def complex(self, k, ind=None, polarization=None, **kwargs):
        '''The complex voltage output can be implemented as a middle step in gain calculation. Can include polarization channels.
        '''
        raise NotImplementedError('')


    def sph_point(self, azimuth, elevation, radians=None):
        '''Point beam towards azimuth and elevation coordinate.

        :param float azimuth: Azimuth east of north of pointing direction.
        :param float elevation: Elevation from horizon of pointing direction.
        :param bool radians: If :code:`True` all input/output angles are in radians, if False degrees are used. Defaults to instance settings :code:`self.radians`.
        :return: :code:`None`

        '''
        azimuth, elevation = self._check_radians(azimuth, elevation, radians)
        sph = Beam._azel_to_numpy(azimuth, elevation)

        self._azimuth = azimuth
        self._elevation = elevation
        self.pointing = coordinates.sph_to_cart(sph, radians = self.radians)


    def point(self, k):
        '''Point beam in local cartesian direction.

        :param numpy.ndarray k: Pointing direction in local coordinates.
        :return: :code:`None`
        '''
        self.pointing = k/np.linalg.norm(k,axis=0)
        sph = coordinates.cart_to_sph(
            self.pointing,
            radians = self.radians,
        )
        self._azimuth = sph[0,...]
        self._elevation = sph[1,...]


    def sph_angle(self, azimuth, elevation, radians=None):
        '''Get angle between azimuth and elevation and pointing direction.

        :param float azimuth: Azimuth east of north to measure from.
        :param float elevation: Elevation from horizon to measure from.
        :param bool radians: If :code:`True` all input/output angles are in radians, if False degrees are used. Defaults to instance settings :code:`self.radians`.

        :return: Angle between pointing and given direction.
        :rtype: float
        '''
        if radians is None:
            radians = self.radians

        sph = Beam._azel_to_numpy(azimuth, elevation)
        k = coordinates.sph_to_cart(sph, radians=radians)
        return self.angle(k, radians=radians)

    def angle(self, k, radians=None):
        '''Get angle between local direction and pointing direction.

        :param numpy.ndarray k: Direction to evaluate angle to.
        :param bool radians: If :code:`True` all input/output angles are in radians, if False degrees are used. Defaults to instance settings :code:`self.radians`.

        :return: Angle between pointing and given direction.
        :rtype: float
        '''
        if radians is None:
            radians = self.radians

        if len(self.pointing.shape) > 1:
            if len(k.shape) > 1:
                theta = np.empty((k.shape[1], self.pointing.shape[1]), dtype=k.dtype)
            else:
                theta = np.empty((self.pointing.shape[1], ), dtype=k.dtype)

            for ind in range(self.pointing.shape[1]):
                theta[...,ind] = coordinates.vector_angle(self.pointing[ind,:], k, radians=radians)
        else:
            theta = coordinates.vector_angle(self.pointing, k, radians=radians)

        return theta


    @abstractmethod
    def gain(self, k, ind=None, polarization=None, **kwargs):
        '''Return the gain in the given direction. This method should be vectorized in the `k` variable.

        If e.g. pointing is the only parameter with 5 directions, :code:`ind=(2,)` would evaluate the gain using the third pointing direction.

        :param numpy.ndarray k: Direction in local coordinates to evaluate gain in. Must be a `(3,)` vector or a `(3,n)` matrix.
        :param tuple ind: The incidences of the available parameters. If the parameters have a size of `1`, no index is needed.
        :param numpy.ndarray polarization: The Jones vector of the incoming plane waves, if applicable for the beam in question.

        :return: Radar gain in the given direction. If input is a `(3,)` vector, output is a float. If input is a `(3,n)` matrix output is a `(n,)` vector of gains.
        :rtype: float/numpy.ndarray
        '''
        pass


    def sph_gain(self, azimuth, elevation, ind=None, polarization=None, radians=None, **kwargs):
        '''Return the gain in the given direction.

        :param float azimuth: Azimuth east of north to evaluate gain in.
        :param float elevation: Elevation from horizon to evaluate gain in.
        :param bool radians: If :code:`True` all input/output angles are in radians, if False degrees are used. Defaults to instance settings :code:`self.radians`.

        :return: Radar gain in the given direction.
        :rtype: float
        '''
        if radians is None:
            radians = self.radians

        sph = Beam._azel_to_numpy(azimuth, elevation)

        k = coordinates.sph_to_cart(sph, radians=radians)
        return self.gain(k, polarization=polarization, ind = ind, **kwargs)

