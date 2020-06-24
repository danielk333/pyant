#!/usr/bin/env python

'''

'''

__all__ = []

from ..gaussian import Gaussian
from ..array import Array
from ..finite_cylindrical_parabola import FiniteCylindricalParabola

from ..beams import Beams

from . import eiscat3d
from . import tromso_space_debris_radar as tsdr_module
from .eiscat_uhf import EISCAT_UHF

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
'''EISCAT 3D Gain pattern for single antenna sub-array.

**Reference:** ???
'''
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
'''EISCAT 3D Gain pattern for a dense core of active sub-arrays, i.e stage 1 of development.

**Reference:** ???
'''
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
'''EISCAT 3D Gain pattern for a full site of active sub-arrays, i.e stage 2 of development.

**Reference:** ???
'''
__all__ += ['e3d_array_stage2']


tsdr = FiniteCylindricalParabola(
    azimuth=0,
    elevation=90.0, 
    frequency=tsdr_module.tsdr_frequency,
    I0=1.0,
    width=120.0,
    height=40.0,
)
'''Tromso Space Debris Radar system with all panels moving as a whole.

**Reference**: ???
'''
__all__ += ['tsdr']

tsdr_fence = [
    FiniteCylindricalParabola(
        azimuth=az,
        elevation=el, 
        frequency=tsdr_module.tsdr_frequency,
        I0=1.0,
        width=30.0,
        height=40.0,
    )
    for az, el in zip([0.0, 0.0, 0.0, 180.0], [30.0, 60.0, 90.0, 60.0])
]
'''Tromso Space Debris Radar system with panels moving independently.

TODO: write proper notes about this: approx with panels and sidelobe distoritons and things
'''
__all__ += ['tsdr_fence']

e_uhf = EISCAT_UHF(
    azimuth=0,
    elevation=90.0,
)
'''EISCAT UHF
'''
__all__ += ['e_uhf']