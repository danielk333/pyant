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


# # Airy disk movie


import matplotlib.pyplot as plt
import numpy as np
import pyant


beam = pyant.models.Airy(
    azimuth=0,
    elevation=90.0,
    frequency=50e6,
    I0=10**4.81,
    radius=10.0,
    degrees=True,
)
num = 100
el = np.linspace(90, 30, num=num)
r = np.linspace(10, 20, num=num)


def update(beam, item):
    beam.parameters["radius"][0] = r[item]
    beam.elevation = el[item]
    return beam


# Note: We need to make sure the animation object is kept in memory by
# saving the return value


# +
fig, ax, mesh, ani = pyant.plotting.gain_heatmap_movie(
    beam,
    iterable=range(num),
    beam_update=update,
    centered=False,
)
plt.show()
