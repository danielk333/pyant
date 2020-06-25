#!/usr/bin/env python

'''

'''

__all__ = []

from ..gaussian import Gaussian
from ..array import Array
from ..finite_cylindrical_parabola import FiniteCylindricalParabola

from . import eiscat3d
from . import tromso_space_debris_radar as tsdr_module
from .tromso_space_debris_radar import find_normalization_constant as tsdr_calibrate
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

**Reference:** [Technical report] Vierinen, J., Kastinen, D., Kero, J., Grydeland, T., McKay, D., Roynestad, E., Hesselbach, S., Kebschull, C., & Krag, H. (2019). EISCAT 3D Performance Analysis
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

**Reference:** [Technical report] Vierinen, J., Kastinen, D., Kero, J., Grydeland, T., McKay, D., Roynestad, E., Hesselbach, S., Kebschull, C., & Krag, H. (2019). EISCAT 3D Performance Analysis
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

**Reference:** [Technical report] Vierinen, J., Kastinen, D., Kero, J., Grydeland, T., McKay, D., Roynestad, E., Hesselbach, S., Kebschull, C., Krag, H. (2019). EISCAT 3D Performance Analysis
'''
__all__ += ['e3d_array_stage2']


tsdr = FiniteCylindricalParabola(
    azimuth=0,
    elevation=90.0, 
    frequency=tsdr_module.tsdr_frequency,
    I0=tsdr_module.tsdr_default_peak_gain,
    width=120.0,
    height=40.0,
)
'''Tromso Space Debris Radar system with all panels moving as a whole.

Has an extra method called :code:`calibrate` that numerically calculates the integral of the gain and scales the gain pattern according.

**Reference**: [White paper] McKay, D., Grydeland, T., Vierinen, J., Kastinen, D., Kero, J., Krag, H. (2019) Conversion of the EISCAT VHF antenna into the Tromso Space Debris Radar
'''
__all__ += ['tsdr']

tsdr_fence = [
    FiniteCylindricalParabola(
        azimuth=az,
        elevation=el, 
        frequency=tsdr_module.tsdr_frequency,
        I0=tsdr_module.tsdr_default_peak_gain/4,
        width=30.0,
        height=40.0,
    )
    for az, el in zip([0.0, 0.0, 0.0, 180.0], [30.0, 60.0, 90.0, 60.0])
]
'''Tromso Space Debris Radar system with panels moving independently.

This model is a list of the 4 panels. This applies heave approximations on the 
behavior of the gain pattern as the panels move. Considering a linefeed of a 
single panel, it will receive more reflection area if one of the adjacent panels
move in into the same pointing direction therefor distorting the side-lobe as 
support structures pass but also narrowing the resulting beam.None of these 
effects are considered here and this approximation is reasonably valid when all 
panels are pointing in sufficiently different directions.

**Reference**: [White paper] McKay, D., Grydeland, T., Vierinen, J., Kastinen, D., Kero, J., Krag, H. (2019) Conversion of the EISCAT VHF antenna into the Tromso Space Debris Radar
'''
__all__ += ['tsdr_fence']


e_uhf = EISCAT_UHF(
    azimuth=0,
    elevation=90.0,
)
'''EISCAT UHF

**Reference**: [Personal communication] Vierinen, J.
'''
__all__ += ['e_uhf']