"""
Halfpipe radar
===========================
"""
import matplotlib.pyplot as plt
import pyant

el = 50

beam = pyant.models.FiniteCylindricalParabola(
    azimuth=0,
    elevation=el,
    frequency=224.0e6,
    width=120.0,
    height=40.0,
    degrees=True,
)

fig, (ax1, ax2) = plt.subplots(1, 2)

pyant.plotting.gain_heatmap(beam, resolution=300, min_elevation=80.0, label="plain", ax=ax1)

# Azimuth is angle (in degrees) clockwise
beam.azimuth = 30.0

pyant.plotting.gain_heatmap(
    beam, resolution=300, min_elevation=80.0, label=f"turned {beam.azimuth} deg clockwise", ax=ax2
)

plt.show()
