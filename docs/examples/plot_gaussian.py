"""
Gaussian planar array gain
===========================
"""

import matplotlib.pyplot as plt
import pyant

beam = pyant.models.Gaussian(
    azimuth=0,
    elevation=90.0,
    frequency=46.5e6,
    I0=10**4.81,
    radius=100.0,
    normal_azimuth=0,
    normal_elevation=90.0,
    degrees=True,
)

fig, (ax1, ax2) = plt.subplots(1, 2)

pyant.plotting.gain_heatmap(beam, resolution=301, min_elevation=80.0, ax=ax1)
ax1.set_title(f"normal-elevation {beam.normal_elevation}")

beam.normal_elevation = 45.0
pyant.plotting.gain_heatmap(beam, resolution=301, min_elevation=80.0, ax=ax2)
ax2.set_title(f"normal-elevation {beam.normal_elevation}")

plt.show()
