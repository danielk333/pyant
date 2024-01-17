#!/usr/bin/env python

"""
Collection of pre-defined radar beams of different Models.

These instances usually correspond to a real physical system.

To register a new radar beam model use the registartor function to pass it the
function that generate the specific radar beam. The function expects a
signiature of only keyword arguments which are options for the generating
function.

"""

# Top level exposed interface
from .beams import beam_of_radar, avalible_beams
from .radar_parameters import parameters_of_radar, avalible_radar_info
from .radar_parameters import UNITS
from ..registry import Radars, Models

# Remember to always import the modules here so that they are executed
# on load of pyant
from . import eiscat_3d
from . import eiscat_uhf
from . import esr
from . import tsdr
from . import arrays
from . import mu
