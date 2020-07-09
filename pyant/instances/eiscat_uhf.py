#!/usr/bin/env python

'''

'''
import pkg_resources

import numpy as np
import scipy.interpolate

from ..beam import Beam
from .. import coordinates

try:
    stream = pkg_resources.resource_stream('pyant.instances.data', 'eiscat_uhf_bp.txt')
    _eiscat_beam_data = np.genfromtxt(stream)
except:
    _eiscat_beam_data = None

class EISCAT_UHF(Beam):
    '''Measured gain pattern of the EISCAT UHF radar.

    **Reference:** ???
    '''

    def __init__(self, azimuth, elevation, frequency = 930e6, **kwargs):
        super().__init__(azimuth, elevation, frequency, **kwargs)

        angle = _eiscat_beam_data[:,0]
        gain = 10**(_eiscat_beam_data[:,1]/10.0)

        self.beam_function = scipy.interpolate.interp1d(np.abs(angle),gain)


    def gain(self, k, polarization=None, ind=None):
        theta = coordinates.vector_angle(self.pointing, k, radians=True)

        sf=self.frequency/930e6 
        G = 10**4.81*self.beam_function(sf*np.abs(theta))

        return G

