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
from .phased_finite_cylindrical_parabola import PhasedFiniteCylindricalParabola
from .interpolated import Interpolation
from .interpolated_array import PlaneArrayInterp
from .validate import validate_functionality

try:
    from . import plotting
except ImportError:
    plotting = None


from .instances import BeamInstancesGetter

instances = BeamInstancesGetter()