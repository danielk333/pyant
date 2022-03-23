'''
Measured beam pattern
======================
'''

import pyant

beam = pyant.beam_of_radar('eiscat_uhf', 'measured')

pyant.plotting.gain_heatmap(beam, min_elevation=80)
pyant.plotting.show()
