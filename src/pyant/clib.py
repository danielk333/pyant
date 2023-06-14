"""python-interface to c-library beam
"""
import sysconfig
import pathlib
import ctypes

# Load the C-lib
suffix = sysconfig.get_config_var("EXT_SUFFIX")
if suffix is None:
    suffix = ".so"

# We start by making a path to the current directory.
pymodule_dir = pathlib.Path(__file__).resolve().parent
__libpath__ = pymodule_dir / ("clibbeam" + suffix)

# Then we open the created shared clibbeam file
clib_beam = ctypes.cdll.LoadLibrary(__libpath__)
