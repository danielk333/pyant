#!/usr/bin/env python

""" """
import types as _t
import importlib.util
from .version import __version__

# Modules
if importlib.util.find_spec("matplotlib") is not None:
    from . import plotting
else:

    class _MissingModule(_t.ModuleType):
        def __getattr__(self, name):
            raise ImportError(
                "The optional dependency `matplotlib` for 'plotting' module is missing.\n"
                "Install it with `pip install pyant[plotting]` or `pip install matplotlib`."
            )

    plotting = _MissingModule("plotting")

from . import models
from . import statistics
from . import beams
from . import types

# from . import clib

from .beam import Beam
