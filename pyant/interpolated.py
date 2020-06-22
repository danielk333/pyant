#!/usr/bin/env python

import numpy as np
import scipy.interpolate
from tqdm import tqdm

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


    def pointing_transform(self, k):
        return k

    def pointing_scale(self, G):
        return G

    def generate_interpolation(self, beam, resolution=1000):

        kx = np.linspace(-1.0, 1.0, num=resolution)
        ky = np.linspace(-1.0, 1.0, num=resolution)
        
        S = np.zeros((resolution,resolution))

        cnt = 0
        tot = resolution**2

        pbar = tqdm(total=tot)
        pbar.set_description('Generating interpolation grid')

        for i,x in enumerate(kx):
            for j,y in enumerate(ky):
                
                pbar.update(1)

                z2 = x**2 + y**2
                if z2 < 1.0:
                    k=np.array([x, y, np.sqrt(1.0 - z2)])
                    S[i,j]=beam.gain(k)
                else:
                    S[i,j] = 0
        pbar.close()
        self.interpolated = scipy.interpolate.interp2d(kx, ky, S.T, kind='linear')


    def save(self, fname):
        np.save(fname, self.interpolated)


    def load(self, fname):
        f_obj = np.load(fname)
        self.interpolated = f_obj.item()
    

    def gain(self, k):
        k_trans = self.pointing_transform(k)

        interp_gain = self.interpolated(k_trans[0], k_trans[1])[0]
        if interp_gain < 0:
            interp_gain = 0.0

        G = self.pointing_scale(interp_gain*self.scaling)

        return G

