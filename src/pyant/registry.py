from enum import Enum


class Models(Enum):
    Airy = 'airy'
    Array = 'array'
    Cassegrain = 'cassegrain'
    Gaussian = 'gaussian'
    Interpolated = 'interpolated'
    InterpolatedArray = 'interpolated_array'
    FiniteCylindricalParabola = 'fcp'
    PhasedFiniteCylindricalParabola = 'phased_fcp'
    Measured = 'measured'


class Radars(Enum):
    EISCAT_3D_module = 'e3d_module'
    EISCAT_3D_stage1 = 'e3d_stage1'
    EISCAT_3D_stage2 = 'e3d_stage2'
    EISCAT_UHF = 'eiscat_uhf'
    ESR_32m = 'esr_32m'
    ESR_42m = 'esr_42m'
    TSDR = 'tsdr'
    MU = 'mu'
    PANSY = 'pansy'
    MAARSY = 'maarsy'
