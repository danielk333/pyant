#!/usr/bin/env python

""" """
import types
from .version import __version__

# Modules
try:
    from . import plotting
except ImportError as err:
    if err.name != "matplotlib":
        raise

    class _MissingModule(types.ModuleType):
        def __getattr__(self, name):
            raise ImportError(
                "The optional dependency `matplotlib` for 'plotting' module is missing.\n"
                "Install it with `pip install pyant[plotting]` or `pip install matplotlib`."
            )

    plotting = _MissingModule("plotting")

from . import coordinates
from . import models
from . import statistics
from . import clib
from . import beams

from .beam import Beam
