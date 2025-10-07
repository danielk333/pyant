# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

raise NotImplementedError()
# # Halfpipe radar

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


# +
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
