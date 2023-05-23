#!/usr/bin/env python

'''

'''
import pkg_resources
import copy

import numpy as np
import scipy.interpolate

from ..models import Measured
from .. import coordinates
from .beams import radar_beam_generator
from ..registry import Radars, Models

try:
    stream = pkg_resources.resource_stream(
        'pyant.beams.data',
        'eiscat_uhf_bp.txt',
    )
    _eiscat_beam_data = np.genfromtxt(stream)
except FileNotFoundError:
    _eiscat_beam_data = None


class EISCAT_UHF(Measured):
    '''Measured gain pattern of the EISCAT UHF radar.

    **Reference:** [Personal communication] Vierinen, J.
    '''

    def __init__(self, azimuth, elevation, frequency = 930e6, **kwargs):
        super().__init__(azimuth, elevation, frequency, **kwargs)

        angle = _eiscat_beam_data[:, 0]
        gain = 10**(_eiscat_beam_data[:, 1]/10.0)

        self.beam_function = scipy.interpolate.interp1d(np.abs(angle), gain)

    def copy(self):
        '''Return a copy of the current instance.
        '''
        return EISCAT_UHF(
            azimuth = copy.deepcopy(self.azimuth),
            elevation = copy.deepcopy(self.elevation),
            frequency = copy.deepcopy(self.frequency),
            radians = self.radians,
        )

    def gain(self, k, polarization=None, ind=None, **kwargs):
        theta = coordinates.vector_angle(self.pointing, k, radians=False)

        sf = self.frequency/930e6
        G = 10**4.81*self.beam_function(sf*np.abs(theta))

        return G


@radar_beam_generator(Radars.EISCAT_UHF, Models.Measured)
def generate_eiscat_uhf_measured():
    return EISCAT_UHF(
        azimuth=0,
        elevation=90.0,
    )
