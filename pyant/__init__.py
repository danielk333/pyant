#!/usr/bin/env python

'''

'''

__version__ = '0.1.0'

from .beam import Beam
from . import coordinates

from .airy import Airy
from .cassegrain import Cassegrain
from .gaussian import Gaussian
from .array import Array
from .unidir_rectangular_dish import UniDirRectangularDish
from .interpolated import Interpolation
from .interpolated_array import PlaneArrayInterp

try:
    from . import plotting
except ImportError:
    plotting = None


from .instances import *