"""
Copy of beams
==============
"""
import matplotlib.pyplot as plt

import pyant

beam = pyant.models.Airy(
    azimuth=0,
    elevation=90.0,
    frequency=[930e6, 220e6],
    I0=10**4.81,
    radius=23.0,
    degrees=True,
)

beam_2 = beam.copy()

beam_2.sph_point(
    azimuth=0,
    elevation=45.0,
)
beam_2.frequency = beam_2.frequency[1:]


fig, (ax1, ax2) = plt.subplots(1, 2)

pyant.plotting.gain_heatmap(beam, ind=(0, 0, 0), resolution=301, min_elevation=80.0, ax=ax1)
pyant.plotting.gain_heatmap(beam_2, ind=(0, 0, 0), resolution=301, min_elevation=80.0, ax=ax2)

plt.show()
