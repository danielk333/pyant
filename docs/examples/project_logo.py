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

# # Make project logo


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import pyant

plt.style.use("dark_background")


el_lim = 1
beam = pyant.models.Cassegrain(
    azimuth=0,
    elevation=90.0,
    frequency=930e6,
    I0=10**4.81,
    inner_radius=23.0,
    outer_radius=40.0,
)
el = np.linspace(90 - el_lim, 90, num=200)
az = np.zeros_like(el)
el = np.append(el, el[::-1])
az = np.append(az, az + 180)


G = beam.sph_gain(az, el, degrees=True)


fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.grid(False)
ax.axis("off")
cmap = mpl.colormaps["winter"]
for scale in np.linspace(0, 1, num=100):
    plt.semilogy(np.arange(G.size), G * scale, c=cmap(scale))
plt.show()
