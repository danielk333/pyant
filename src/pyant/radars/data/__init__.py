"""
Package for all the data files needed to instantiate a description of a radar station.

References for the data:

#TODO

"""
import json
import pathlib
import importlib.resources

DATA = {}

# To be compatible with 3.7-8
# as resources.files was introduced in 3.9
if hasattr(importlib.resources, "files"):
    _data_files = importlib.resources.files("pyant.radars.data")
    for file in _data_files.iterdir():
        if not file.is_file():
            continue
        if file.name.endswith(".py"):
            continue

        DATA[file.name] = file

else:
    _data_folder = importlib.resources.contents("pyant.radars.data")
    for fname in _data_folder:
        with importlib.resources.path("pyant.radars.data", fname) as file:
            if not file.is_file():
                continue
            if file.name.endswith(".py"):
                continue

            DATA[file.name] = pathlib.Path(str(file))



RADAR_PARAMETERS = dict()

# Load parameters
_data_file = DATA["radar_parameters.json"] if "radar_parameters.json" in DATA else None
if _data_file is not None:
    with _data_file.open("r") as stream:
        RADAR_PARAMETERS = json.loads(stream.read())
