'''
Omni-directional antenna
========================
'''

import pyant
import pyant.instances as lib

pyant.plotting.gain_heatmap(lib.e_uhf)
pyant.plotting.show()