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

# # Gaussian planar array gain

import numpy as np
import matplotlib.pyplot as plt
import pyant


beam = pyant.models.Gaussian(
    pointing=np.array([0, 0, 1], dtype=np.float64),
    frequency=46.5e6,
    radius=100.0,
    normal_pointing=np.array([0, 0, 1], dtype=np.float64),
    peak_gain=10**4.81,
)

# +
# # Plot the gain pattern in zenith for two different planar tilts
fig, (ax1, ax2) = plt.subplots(1, 2)
pyant.plotting.gain_heatmap(beam, resolution=301, min_elevation=80.0, ax=ax1)
ax1.set_title("Plane normal-elevation = 90 deg")

beam.parameters["normal_pointing"] = pyant.coordinates.sph_to_cart(
    np.array([0, 45.0, 1]), degrees=True
)

pyant.plotting.gain_heatmap(beam, resolution=301, min_elevation=80.0, ax=ax2)
ax2.set_title("Plane normal-elevation = 45 deg")


# +
# # Plot the gain 1 degree off bore-sight as a function of frequency and radius
# This is an example of setting up a larger vectorized calculation using `pyant`
# and then evaluating that calculation and formatting it back to its intended
# shape and parameter space

num = 100
tnum = num**2
freq = np.linspace(50e6, 900e6, num)
radius = np.linspace(23, 100, num)
fmat, rmat = np.meshgrid(freq, radius)

beam.parameters["normal_pointing"] = np.zeros((3, tnum), dtype=np.float64)
beam.parameters["normal_pointing"][2, :] = 1
beam.parameters["pointing"] = beam.parameters["normal_pointing"]
beam.parameters["frequency"] = fmat.reshape((tnum,))
beam.parameters["radius"] = rmat.reshape((tnum,))

k_in = pyant.coordinates.sph_to_cart(np.array([0, 89.0, 1]), degrees=True)

g = beam.gain(k_in)

fig, ax = plt.subplots()
pm = ax.pcolormesh(fmat * 1e-6, rmat, 10 * np.log10(g.reshape(fmat.shape)))
ax.set_xlabel("Frequency [MHz]")
ax.set_ylabel("Radius [m]")
ax.set_title("1-degree off bore-sight gain")
cb = fig.colorbar(pm, ax=ax)
cb.set_label("Gain [dB]")

plt.show()
