#!/usr/bin/env python

import functools
import operator as op

from .beam import Beam
from . import coordinates

class Beams(list):
    '''Collection of beams.

    Creates super-methods prefixed with :code:`get_` and :code:`total_`. 
    The :code:`get_X` methods calls the method :code:`X` on each element 
    in the list and returns a list with the results from all beams. 
    The :code:`total_X` method calls the :code:`get_X` method and performs 
    `reduce` on the result using the supplied `operator`. 
    `operator` Can be a instance retrieved from the `operator` module, 
    e.g :code:`operator='add'` to perform a sum.

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
