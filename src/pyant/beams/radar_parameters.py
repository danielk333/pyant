import json
import pkg_resources

from ..registry import Radars

RADAR_PARAMETERS = dict()

# Load parameters

__data = pkg_resources.resource_string('pyant.beams.data', 'radar_parameters.json')
__data = json.loads(__data.decode('utf-8'))
for key in __data:
    RADAR_PARAMETERS[Radars(key)] = __data[key]
del __data


def parameters_of_radar(radar):
    try:
        radar_item = Radars(radar)
    except ValueError:
        raise ValueError(
            f'"{radar}" radar not found. See avalible Radars:\n'
            + ', '.join([str(x) for x in Radars])
        )

    if radar_item not in RADAR_PARAMETERS:
        raise ValueError(
            f'No recorded parameters for radar "{radar_item}". See avalible Radars:\n'
            + ', '.join([str(x) for x in RADAR_PARAMETERS])
        )

    return RADAR_PARAMETERS[radar_item]
