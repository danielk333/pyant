'''
TSDR
=====
'''
import numpy as np

import pyant
import pyant.instances as lib

lib.tsdr.frequency = 1.2e9
for panel in lib.tsdr_fence:
    panel.frequency = 1.2e9

print(f'TSDR Normalization constant {lib.tsdr.I0}')
lib.tsdr.I0 = lib.tsdr_calibrate(lib.tsdr)
print(f'TSDR Normalization constant post calibration {lib.tsdr.I0}')

fig, ax = pyant.plotting.gain_heatmap(lib.tsdr, resolution=300, min_elevation=75.0)
ax.set_title('TSDR', fontsize=22)

print(f'TSDR Panel Normalization constant {[panel.I0 for panel in lib.tsdr_fence]}')
panel_I0 = lib.tsdr_calibrate(lib.tsdr_fence[2])
for panel in lib.tsdr_fence:
    panel.I0 = panel_I0


print(f'TSDR Panel Normalization constant post calibration {[panel.I0 for panel in lib.tsdr_fence]}')

fig, ax = pyant.plotting.gain_heatmap(lib.tsdr_fence, resolution=600, min_elevation=0)

pyant.plotting.show()

