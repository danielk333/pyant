#!/usr/bin/env python

'''

'''

from .version import __version__

from .beam import Beam
from . import coordinates

from .airy import Airy
from .cassegrain import Cassegrain
from .gaussian import Gaussian
from .array import Array, DipoleArray
from .finite_cylindrical_parabola import FiniteCylindricalParabola
from .interpolated import Interpolation
from .interpolated_array import PlaneArrayInterp

try:
    from . import plotting
except ImportError:
    plotting = None


from . import instances

def __getattr__(name):
    if name in instances.beam_instances:
        return getattr(instances, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
