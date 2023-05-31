import importlib

_data_folder = importlib.resources.files("pyant.beams.data")
DATA = {}
for file in _data_folder.iterdir():
    if not file.is_file():
        continue
    if file.name.endswith(".py"):
        continue

    DATA[file.name] = file
