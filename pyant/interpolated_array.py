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


    def pointing_transform(self, k, pointing, polarization=None):
        k_ = k/np.linalg.norm(k, axis=0)

        theta = -np.arctan2(pointing[1], pointing[0])
        
        M_rot = coordinates.rot2d(theta)
        M_scale = coordinates.scale2d(pointing[2], 1)
        M_rot_inv = coordinates.rot2d(-theta)

        M = M_rot_inv.dot(M_scale.dot(M_rot))
        
        k_trans = np.empty(k_.shape, dtype = np.float)
        k_[0,...] -= pointing[0]
        k_[1,...] -= pointing[1]
        k_trans[:2,...] = M.dot(k_[:2,...])
        k_trans[2,...] = k_[2,...]
        return k_trans


    def pointing_scale(self, G, pointing):
        return G*pointing[2]
