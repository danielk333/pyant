'''
Predefined instances
======================
'''
import numpy as np

import pyant
import pyant.instances as lib

print(lib.__all__)

for inst in lib.__all__:
    fig, ax = pyant.plotting.gain_heatmap(getattr(lib, inst), resolution=100, min_elevation=45.0)
    ax.set_title(inst.replace('_', ' '))

pyant.plotting.show()