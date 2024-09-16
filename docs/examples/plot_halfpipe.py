"""
Halfpipe radar
===========================
"""
import matplotlib.pyplot as plt
import pyant

el = 90

beam = pyant.models.FiniteCylindricalParabola(
    azimuth=0,
    elevation=el,
    frequency=224.0e6,
    width=120.0,
    height=40.0,
    degrees=True,
)

fig, (ax1, ax2) = plt.subplots(1, 2)

pyant.plotting.gain_heatmap(
    beam,
    resolution=300,
    min_elevation=80.0,
    ax=ax1,
)

beam.sph_point(0, 30)

pyant.plotting.gain_heatmap(
    beam,
    resolution=300,
    min_elevation=80.0,
    label=f" - pointed {beam.elevation} deg elevation",
    ax=ax2,
)

plt.show()
