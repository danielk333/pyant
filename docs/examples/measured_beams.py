"""
Measured beam pattern
======================
"""
import matplotlib.pyplot as plt
import pyant

beam = pyant.beam_of_radar("eiscat_uhf", "measured")

fig, (ax1, ax2) = plt.subplots(1, 2)

pyant.plotting.gain_heatmap(beam, min_elevation=80, ax=ax1)
pyant.plotting.gains(beam, min_elevation=80, ax=ax2)

plt.show()
