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

# # Airy disk antenna gain

import numpy as np
import matplotlib.pyplot as plt
import pyant


beam = pyant.models.Airy(
    peak_gain=10**4.81,
)
param = pyant.models.AiryParams(
    pointing=np.array([0, 0, 1], dtype=np.float64),
    frequency=930e6,
    radius=23.0,
)
beam_c = pyant.models.Cassegrain(
    peak_gain=10**4.81,
)
param_c = pyant.models.CassegrainParams(
    pointing=np.array([0, 0, 1], dtype=np.float64),
    frequency=930e6,
    outer_radius=40.0,
    inner_radius=23.0,
)

# +
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
pyant.plotting.gain_heatmap(
    beam,
    param,
    resolution=301,
    min_elevation=87.0,
    ax=ax1,
    cbar_min=0,
)
ax1.set_title("Airy")

pyant.plotting.gain_heatmap(
    beam_c,
    param_c,
    resolution=301,
    min_elevation=87.0,
    ax=ax2,
    cbar_min=0,
)
ax2.set_title("Cassegrain")
plt.show()
