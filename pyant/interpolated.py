#!/usr/bin/env python
import copy

import numpy as np
import scipy.interpolate

from .beam import Beam
from . import coordinates

class Interpolation(Beam):
    '''Interpolated gain pattern. Does not assume any effect on pointing.

    :param float scaling: Scaling of the gain pattern to apply.

    :ivar float scaling: Scaling of the gain pattern to apply.
    '''
    def __init__(self, azimuth, elevation, frequency, scaling = 1.0, **kwargs):
        super().__init__(azimuth, elevation, frequency, **kwargs)
        self.interpolated = None
        self.scaling = scaling


    def copy(self):
        '''Return a copy of the current instance.
        '''
        bm = Interpolation(
            azimuth = copy.deepcopy(self.azimuth),
            elevation = copy.deepcopy(self.elevation),
            frequency = copy.deepcopy(self.frequency),
            scaling = copy.deepcopy(self.scaling),
            radians = self.radians,
        )
        bm.interpolated = self.interpolated
        return bm


    def pointing_transform(self, k, pointing, polarization=None):
        return k

    def pointing_scale(self, G, pointing, polarization=None):
        return G

    def generate_interpolation(self, beam, resolution=1000):

        kx = np.linspace(-1.0, 1.0, num=resolution)
        ky = np.linspace(-1.0, 1.0, num=resolution)
        
        size = resolution**2

        xv, yv = np.meshgrid(kx, ky, sparse=False, indexing='ij')
        k = np.empty((3,size), dtype=np.float64)
        k[0,:] = xv.reshape(1,size)
        k[1,:] = yv.reshape(1,size)
        z2 = k[0,:]**2 + k[1,:]**2

        inds_ = z2 <= 1.0
        not_inds_ = np.logical_not(inds_)

        k[2,inds_] = np.sqrt(1.0 - z2[inds_])

        G = np.zeros((1,size))
        G[0,inds_] = beam.gain(k[:,inds_])
        G = G.reshape(len(kx),len(ky))

        self.interpolated = scipy.interpolate.RectBivariateSpline(kx, ky, G.T)


    def save(self, fname):
        np.save(fname, self.interpolated)


    def load(self, fname):
        f_obj = np.load(fname, allow_pickle=True)
        self.interpolated = f_obj.item()
    

    def gain(self, k, polarization=None, ind=None):
        pointing, frequency = self.get_parameters(ind)
        k_trans = self.pointing_transform(k, pointing)

        interp_gain = self.interpolated(k_trans[0,...], k_trans[1,...], grid=False)
        interp_gain[interp_gain < 0] = 0

        if len(k.shape) == 1:
            if len(interp_gain.shape) > 0:
                interp_gain = interp_gain[0]

        G = self.pointing_scale(interp_gain*self.scaling, pointing)

        return G

