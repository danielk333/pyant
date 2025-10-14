"""
Package for all the data files needed to instantiate a description of a radar station.

References for the data:

#TODO
"""

import pathlib
import importlib.resources

DATA_FILES = {}

# To be compatible with 3.7-8
# as resources.files was introduced in 3.9
if hasattr(importlib.resources, "files"):
    _data_files = importlib.resources.files("pyant.radars.data")
    for file in _data_files.iterdir():
        if not file.is_file():
            continue
        if file.name.endswith(".py"):
            continue

        DATA_FILES[file.name] = file

else:
    _data_folder = importlib.resources.contents("pyant.radars.data")
    for fname in _data_folder:
        with importlib.resources.path("pyant.radars.data", fname) as file:
            if not file.is_file():
                continue
            if file.name.endswith(".py"):
                continue

            DATA_FILES[file.name] = pathlib.Path(str(file))
