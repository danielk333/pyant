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
import spacecoords.spherical as sph
import pyant


beam = pyant.models.Airy(
    peak_gain=10**4.81,
)
param = pyant.models.AiryParams(
    pointing=np.array([0, 0, 1], dtype=np.float64),
    frequency=50e6,
    radius=10.0,
)
num = 100
el = np.concatenate(
    [
        np.linspace(90, 30, num=num),
        np.linspace(30, 90, num=num),
    ]
)
r = np.linspace(10, 20, num=num * 2)


def update(param, item):
    param.radius = r[item]
    param.pointing = sph.az_el_point(azimuth=0, elevation=el[item], degrees=True)
    return param


# Note: We need to make sure the animation object is kept in memory by
# saving the return value


# +
fig, ax, mesh, ani = pyant.plotting.gain_heatmap_movie(
    beam,
    param,
    iterable=range(num * 2),
    param_update=update,
    centered=False,
    fps=40,
)
plt.show()
