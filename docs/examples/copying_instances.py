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

# # Copying beam instances

import matplotlib.pyplot as plt
import numpy as np

import pyant

# Lets create a simple Airy disk model and copy it

beam = pyant.models.Airy(
    pointing=np.array([0, 0, 1], dtype=np.float64),
    frequency=500e6,
    radius=23.0,
    peak_gain=10**4.81,
)

beam_2 = beam.copy()

# These two copies are now independant

beam_2.sph_point(
    azimuth=45.0,
    elevation=45.0,
    degrees=True,
)
beam_2.frequency = 50e6


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
pyant.plotting.gain_heatmap(beam, resolution=301, min_elevation=80.0, ax=ax1)
pyant.plotting.gain_heatmap(beam_2, resolution=301, min_elevation=80.0, ax=ax2)
plt.show()
