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

# # Surface plot


import matplotlib.pyplot as plt
import pyant


beam = pyant.models.Cassegrain(
    azimuth=0,
    elevation=90.0,
    frequency=930e6,
    I0=10**4.81,
    outer_radius=40.0,
    inner_radius=23.0,
    degrees=True,
)


# +
pyant.plotting.gain_surface(beam, resolution=301, min_elevation=85.0)
plt.show()
