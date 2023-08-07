import json

from ..registry import Radars
from .data import DATA

RADAR_PARAMETERS = dict()

# Load parameters

_data_file = DATA["radar_parameters.json"] if "radar_parameters.json" in DATA else None
if _data_file is not None:
    with _data_file.open("r") as stream:
        __data = json.loads(stream.read())
    for key in __data:
        RADAR_PARAMETERS[Radars(key)] = __data[key]
    del __data

# Define units

UNITS = {
    "frequency": "Hz",
    "power": "W",
    "lat": "deg",
    "lon": "deg",
    "alt": "m",
}


def avalible_radar_info():
    """Returns a dict listing all avalible Radars and their Models"""
    return {key: list(val.keys()) for key, val in RADAR_PARAMETERS.items()}


def parameters_of_radar(radar):
    try:
        radar_item = Radars(radar)
    except ValueError:
        raise ValueError(
            f'"{radar}" radar not found. See avalible Radars:\n'
            + ", ".join([str(x) for x in Radars])
        )

    if radar_item not in RADAR_PARAMETERS:
        raise ValueError(
            f'No recorded parameters for radar "{radar_item}". See avalible Radars:\n'
            + ", ".join([str(x) for x in RADAR_PARAMETERS])
        )

    return RADAR_PARAMETERS[radar_item]
