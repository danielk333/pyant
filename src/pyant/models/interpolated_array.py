#!/usr/bin/env python
import copy

from ..interpolated import Interpolated


class InterpolatedArray(Interpolated):
    '''Interpolated gain pattern of an planar antenna array.
    Translates and scales the interpolated gain pattern to the pointing direction.
    '''

    def __init__(self, azimuth, elevation, frequency, **kwargs):
        super().__init__(azimuth, elevation, frequency, **kwargs)

    def copy(self):
        '''Return a copy of the current instance.'''
        bm = InterpolatedArray(
            azimuth=copy.deepcopy(self.azimuth),
            elevation=copy.deepcopy(self.elevation),
            frequency=copy.deepcopy(self.frequency),
            scaling=copy.deepcopy(self.scaling),
            radians=self.radians,
        )
        bm.interpolated = self.interpolated
        return bm

    def pointing_transform(self, k, pointing, polarization=None):
        return k - pointing[:, None]

    def pointing_scale(self, k, pointing, G):
        return G * k[2]
