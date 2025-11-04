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

# # Random arrays

import matplotlib.pyplot as plt
import numpy as np
import pyant

np.random.seed(1234)
ant_n = 300


antennas = np.zeros((3, ant_n, 1))
antennas[:2, :, 0] = np.random.rand(2, ant_n) * 50 - 25
beam = pyant.models.Array(
    antennas=antennas,
)
param = pyant.models.ArrayParams(
    pointing=np.array([0, 0, 1], dtype=np.float64),
    frequency=46.5e6,
    polarization=beam.polarization.copy(),
)


fig, (ax1, ax2) = plt.subplots(1, 2)
pyant.plotting.gain_heatmap(beam, param, resolution=300, min_elevation=60.0, ax=ax1)
pyant.plotting.antenna_configuration(beam.antennas, z_axis=False, ax=ax2)
plt.show()
