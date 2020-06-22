#!/usr/bin/env python

import numpy as np
from tqdm import tqdm

from .interpolated import Interpolation
from . import coordinates


class PlaneArrayInterp(Interpolation):
    '''Interpolated gain pattern of an planar antenna array. Translates and scales the interpolated gain pattern to the pointing direction.

    '''
    def __init__(self, azimuth, elevation, frequency, **kwargs):
        super().__init__(azimuth, elevation, frequency, **kwargs)


    def pointing_transform(self, k):
        k_ = k/np.linalg.norm(k)

        theta = -np.arctan2(self.pointing[1], self.pointing[0])
        
        M_rot = coordinates.rot2d(theta)
        M_scale = coordinates.scale2d(self.pointing[2], 1)
        M_rot_inv = coordinates.rot2d(-theta)

        M = M_rot_inv.dot(M_scale.dot(M_rot))
        
        k_trans = np.empty((3,), dtype = np.float)
        k_trans[:2] = M.dot(k_[:2] - self.pointing[:2])
        k_trans[2] = k_[2]
        return k_trans


    def pointing_scale(self, G):
        return G*self.pointing[2]
