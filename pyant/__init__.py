#!/usr/bin/env python

'''

'''

__version__ = '0.1.0'



from .beam import Beam
from . import coordinates

from .airy import Airy

try:
    from . import plotting
except ImportError:
    plotting = None
