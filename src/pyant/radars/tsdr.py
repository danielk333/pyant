#!/usr/bin/env python

"""A collection of functions and information for the TSDR Radar system.

Notes
-----
Configurations taken from [1]_.

.. [1] (White paper) McKay, D., Grydeland, T., Vierinen, J.,
    Kastinen, D., Kero, J., Krag, H. (2019) Conversion of the EISCAT VHF
    antenna into the Tromso Space Debris Radar

"""

from ..models import FiniteCylindricalParabola
from ..models import PhasedFiniteCylindricalParabola
from .beams import radar_beam_generator
from ..registry import Radars, Models


@radar_beam_generator(Radars.TSDR, Models.FiniteCylindricalParabola)
def generate_tsdr():
    """Tromso Space Debris Radar system with all panels moving as a whole [1]_.

    Notes
    -----
    Has an extra method called :code:`calibrate` that numerically calculates
    the integral of the gain and scales the gain pattern according.



    """
    return FiniteCylindricalParabola(
        azimuth=0,
        elevation=90.0,
        frequency=1.8e9,
        I0=None,
        width=120.0,
        height=40.0,
        degrees=True,
    )


@radar_beam_generator(Radars.TSDR, Models.PhasedFiniteCylindricalParabola)
def generate_tsdr_phased():
    """Tromso Space Debris Radar system with panels moving independently.

    Notes
    -----
    This model is a list of the 4 panels. This applies heave approximations on
    the behavior of the gain pattern as the panels move. Considering a linefeed
    of a single panel, it will receive more reflection area if one of the
    adjacent panels move in into the same pointing direction therefor
    distorting the side-lobe as support structures pass but also narrowing the
    resulting beam.None of these effects are considered here and this
    approximation is reasonably valid when all panels are pointing in
    sufficiently different directions.

    """
    return PhasedFiniteCylindricalParabola(
        azimuth=0,
        elevation=90.0,
        phase_steering=0.0,
        depth=18.0,
        frequency=1.8e9,
        I0=None,
        width=120.0,
        height=40.0,
        degrees=True,
    )
