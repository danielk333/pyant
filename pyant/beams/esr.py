#!/usr/bin/env python

'''A collection of functions and information for the TSDR Radar system.

'''


from .. import Cassegrain
from .beams import radar_beam_generator


@radar_beam_generator('esr_32m', 'cassegrain')
def generate_esr_32m():
    '''ESR

    '''
    return Cassegrain(
        0.0,              # azimuth
        90.0,             # elevation
        500e6,            # frequency
        10**(42.5 / 10),  # Linear gain (42.5 dB)
        16.0,             # radius main reflector
        1.73,             # radius subreflector (eyeballed from photo)
    )


@radar_beam_generator('esr_42m', 'cassegrain')
def generate_esr_42m():
    '''ESR

    '''
    return Cassegrain(
        185.5,            # azimuth     (since 2019)
        82.1,             # elevation   (since 2019)
        500e6,            # frequency
        10**(45.0 / 10),  # Linear gain (42.5 dB)
        21.0,             # radius main reflector
        3.3,              # radius subreflector (eyeballed from photo)
    )
