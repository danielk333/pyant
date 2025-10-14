#!/usr/bin/env python

"""
Collection of pre-defined radars.

These instances usually correspond to a real physical system.
"""

# imports of all available radar generators
from .eiscat_uhf import (
    generate_eiscat_uhf_tromso,
    generate_eiscat_uhf_kiruna,
    generate_eiscat_uhf_sodankyla,
)
# from .mu import generate_mu_radar
