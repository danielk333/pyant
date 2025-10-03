import json

from .data import DATA

RADAR_PARAMETERS = dict()

# Load parameters
_data_file = DATA["radar_parameters.json"] if "radar_parameters.json" in DATA else None
if _data_file is not None:
    with _data_file.open("r") as stream:
        RADAR_PARAMETERS = json.loads(stream.read())

# Define units
UNITS = {
    "frequency": "Hz",
    "power": "W",
    "lat": "deg",
    "lon": "deg",
    "alt": "m",
}
