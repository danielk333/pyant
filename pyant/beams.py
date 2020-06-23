#!/usr/bin/env python

import functools
import operator as op

from .beam import Beam
from . import coordinates

class Beams(list):
    '''Collection of beams.
    '''
    @property
    def wavelength(self):
        return [b.wavelength for b in self]


def _add_super_methods(method):
    def list_method(self, *args, **kwargs):
        return [getattr(beam, method)(*args, **kwargs) for beam in self]

    setattr(Beams, 'get_' + method, list_method)

    def total_method(self, operator, *args, **kwargs):
        lst = getattr(self, 'get_' + method)(*args, **kwargs)
        return functools.reduce(getattr(op, operator), lst)

    setattr(Beams, 'total_' + method, total_method)


_add_super_methods('gain')
_add_super_methods('sph_gain')
_add_super_methods('angle')
_add_super_methods('sph_angle')
_add_super_methods('point')
_add_super_methods('sph_point')
