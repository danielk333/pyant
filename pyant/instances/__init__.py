#!/usr/bin/env python

'''

'''



from ..gaussian import Gaussian
from ..array import Array
from ..finite_cylindrical_parabola import FiniteCylindricalParabola
from ..phased_finite_cylindrical_parabola import PhasedFiniteCylindricalParabola
from ..cassegrain import Cassegrain

from . import eiscat3d
from .eiscat_uhf import EISCAT_UHF

__all__ = []
beam_instances = [
    'e3d_array_module',
    'e3d_array_stage1',
    'e3d_array_stage2',
    'tsdr',
    'tsdr_phased',
    'e_uhf',
    'esr_32',
    'esr_42',
]

class BeamInstancesGetter:
    '''

    :e3d_array_module:

        EISCAT 3D Gain pattern for single antenna sub-array.

        **Reference:** [Technical report] Vierinen, J., Kastinen, D., Kero, J., Grydeland, T., McKay, D., Roynestad, E., Hesselbach, S., Kebschull, C., & Krag, H. (2019). EISCAT 3D Performance Analysis


    :e3d_array_stage1:

        EISCAT 3D Gain pattern for a dense core of active sub-arrays, i.e stage 1 of development.

        **Reference:** [Technical report] Vierinen, J., Kastinen, D., Kero, J., Grydeland, T., McKay, D., Roynestad, E., Hesselbach, S., Kebschull, C., & Krag, H. (2019). EISCAT 3D Performance Analysis


    :e3d_array_stage2:

        EISCAT 3D Gain pattern for a full site of active sub-arrays, i.e stage 2 of development.

        **Reference:** [Technical report] Vierinen, J., Kastinen, D., Kero, J., Grydeland, T., McKay, D., Roynestad, E., Hesselbach, S., Kebschull, C., & Krag, H. (2019). EISCAT 3D Performance Analysis


    :tsdr:

        Tromso Space Debris Radar system with all panels moving as a whole.

        Has an extra method called :code:`calibrate` that numerically calculates the integral of the gain and scales the gain pattern according.

        **Reference**: [White paper] McKay, D., Grydeland, T., Vierinen, J., Kastinen, D., Kero, J., Krag, H. (2019) Conversion of the EISCAT VHF antenna into the Tromso Space Debris Radar


    :tsdr_fence:


        Tromso Space Debris Radar system with panels moving independently.

        This model is a list of the 4 panels. This applies heave approximations on the 
        behavior of the gain pattern as the panels move. Considering a linefeed of a 
        single panel, it will receive more reflection area if one of the adjacent panels
        move in into the same pointing direction therefor distorting the side-lobe as 
        support structures pass but also narrowing the resulting beam.None of these 
        effects are considered here and this approximation is reasonably valid when all 
        panels are pointing in sufficiently different directions.

        **Reference**: [White paper] McKay, D., Grydeland, T., Vierinen, J., Kastinen, D., Kero, J., Krag, H. (2019) Conversion of the EISCAT VHF antenna into the Tromso Space Debris Radar


    :e_uhf:

        EISCAT UHF measured beam pattern.

        **Reference**: [Personal communication] Vierinen, J.


    '''

    instances = beam_instances
    __all__ = beam_instances

    def __getattr__(self, name):

        if name == 'e3d_array_module':
            return Array(
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
            
        elif name == 'e3d_array_stage1':
            return Array(
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

            
        elif name == 'e3d_array_stage2':
            return Array(
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
            
        elif name == 'tsdr':
            return FiniteCylindricalParabola(
                azimuth=0,
                elevation=90.0, 
                frequency=1.8e9,
                I0=None,
                width=120.0,
                height=40.0,
            )

        elif name == 'tsdr_phased':
            return PhasedFiniteCylindricalParabola(
                azimuth=0,
                elevation=90.0,
                phase_steering = 0.0,
                depth=18.0,
                frequency=1.8e9,
                I0=None,
                width=120.0,
                height=40.0,
            )


        elif name == 'e_uhf':
            return EISCAT_UHF(
                azimuth=0,
                elevation=90.0,
            )

        elif name == 'esr_32' :
            return Cassegrain(
                0.0,              # azimuth
                90.0,             # elevation
                500e6,            # frequency
                10**(42.5 / 10),  # Linear gain (42.5 dB)
                16.0,             # radius longitudinal direction
                16.0,             # radius latitudinal direction
            )

        elif name == 'esr_42' :
            return Cassegrain(
                185.5,            # azimuth     (since 2019)
                82.1,             # elevation   (since 2019)
                500e6,            # frequency
                10**(45.0 / 10),  # Linear gain (42.5 dB)
                21.0,             # radius longitudinal direction
                21.0,             # radius latitudinal direction
            )

        raise AttributeError(f"module '{__name__}' has no attribute '{name}'\n(valid names are {beam_instances})")

