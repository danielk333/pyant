#!/usr/bin/env python

'''

'''

__all__ = []

from ..gaussian import Gaussian
from ..array import Array

from . import eiscat3d


e3d_array_module = Array(
    azimuth = 0.0, 
    elevation = 90.0, 
    frequency = eiscat3d.e3d_frequency, 
    antennas = eiscat3d.e3d_array(
        eiscat3d.e3d_frequency,
        configuration='module',
    ), 
    scaling = eiscat3d.e3d_antenna_gain,
    radians = False,
)
__all__ += ['e3d_array_module']


e3d_array_stage1 = Array(
    azimuth = 0.0, 
    elevation = 90.0, 
    frequency = eiscat3d.e3d_frequency, 
    antennas = eiscat3d.e3d_array(
        eiscat3d.e3d_frequency,
        configuration='half-dense',
    ), 
    scaling = eiscat3d.e3d_antenna_gain,
    radians = False,
)
__all__ += ['e3d_array_stage1']


e3d_array_stage2 = Array(
    azimuth = 0.0, 
    elevation = 90.0, 
    frequency = eiscat3d.e3d_frequency, 
    antennas = eiscat3d.e3d_array(
        eiscat3d.e3d_frequency,
        configuration='full',
    ), 
    scaling = eiscat3d.e3d_antenna_gain,
    radians = False,
)
__all__ += ['e3d_array_stage2']