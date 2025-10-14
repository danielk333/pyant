#!/usr/bin/env python

""" """
import types
import importlib
from .version import __version__

# Modules
if importlib.util.find_spec("matplotlib") is not None:
    from . import plotting
else:

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
from . import beams

# from . import clib

from .beam import Beam
