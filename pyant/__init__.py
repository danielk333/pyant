#!/usr/bin/env python

'''GAIN IS AMPLITUDE AMPLIFICATION!!!!! DAMIT 

'''

from .version import __version__

from .beam import Beam
from . import coordinates

from .airy import Airy
from .cassegrain import Cassegrain
from .gaussian import Gaussian
from .array import Array
from .finite_cylindrical_parabola import FiniteCylindricalParabola
from .interpolated import Interpolation
from .interpolated_array import PlaneArrayInterp

try:
    from . import plotting
except ImportError:
    plotting = None


from .instances import *