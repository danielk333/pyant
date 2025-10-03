#!/usr/bin/env python

"""A collection of functions and information for the TSDR Radar system.

"""


from ..models import Cassegrain
from .beams import radar_beam_generator
from ..registry import Radars, Models


@radar_beam_generator(Radars.ESR_32m, Models.Cassegrain)
def generate_esr_32m():
    """ESR"""
    return Cassegrain(
        0.0,  # azimuth
        90.0,  # elevation
        500e6,  # frequency
        10 ** (42.5 / 10),  # Linear gain (42.5 dB)
        16.0,  # radius main reflector
        1.73,  # radius subreflector (eyeballed from photo)
        degrees=True,
    )


@radar_beam_generator(Radars.ESR_42m, Models.Cassegrain)
def generate_esr_42m():
    """ESR"""
    return Cassegrain(
        185.5,  # azimuth     (since 2019)
        82.1,  # elevation   (since 2019)
        500e6,  # frequency
        10 ** (45.0 / 10),  # Linear gain (42.5 dB)
        21.0,  # radius main reflector
        3.3,  # radius subreflector (eyeballed from photo)
        degrees=True,
    )
