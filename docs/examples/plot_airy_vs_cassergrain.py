"""
Airy disk antenna gain
========================
"""
import matplotlib.pyplot as plt
import pyant

beam = pyant.models.Airy(
    azimuth=0,
    elevation=90.0,
    frequency=930e6,
    I0=10**4.81,
    radius=23.0,
    degrees=True,
)

beam_c = pyant.models.Cassegrain(
    azimuth=0,
    elevation=90.0,
    frequency=930e6,
    I0=10**4.81,
    outer_radius=40.0,
    inner_radius=23.0,
    degrees=True,
)

fig, (ax1, ax2) = plt.subplots(1, 2)

pyant.plotting.gain_heatmap(beam, resolution=301, min_elevation=85.0, ax=ax1)
pyant.plotting.gain_heatmap(beam_c, resolution=301, min_elevation=85.0, ax=ax2)

plt.show()
