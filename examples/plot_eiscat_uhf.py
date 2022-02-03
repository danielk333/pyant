'''
EISCAT UHF
======================
'''
import numpy as np

import pyant
import pyant.instances as lib


beam = lib.e_uhf

pyant.plotting.gain_heatmap(beam, resolution=100, min_elevation=80.0)
pyant.plotting.gains(beam, resolution=1000, min_elevation = 80.0)

pyant.plotting.show()