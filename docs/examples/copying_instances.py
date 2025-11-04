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
    peak_gain=10**4.81,
)
param = pyant.models.AiryParams(
    pointing=np.array([0, 0, 1], dtype=np.float64),
    frequency=500e6,
    radius=23.0,
)
beam_2 = beam.copy()
param_2 = param.copy()

# These copies are now independent

beam_2.peak_gain = 10**3.8
param_2.radius = 10.0

fig, axes = plt.subplots(2, 2, figsize=(12, 5))
pyant.plotting.gain_heatmap(
    beam,
    param,
    resolution=301,
    min_elevation=80.0,
    ax=axes[0, 0],
    cbar_min=0,
    cbar_max=48.1,
    label="beam & param",
)
pyant.plotting.gain_heatmap(
    beam_2,
    param,
    resolution=301,
    min_elevation=80.0,
    ax=axes[0, 1],
    cbar_min=0,
    cbar_max=48.1,
    label="beam_2 & param",
)
pyant.plotting.gain_heatmap(
    beam,
    param_2,
    resolution=301,
    min_elevation=80.0,
    ax=axes[1, 0],
    cbar_min=0,
    cbar_max=48.1,
    label="beam & param_2",
)
pyant.plotting.gain_heatmap(
    beam_2,
    param_2,
    resolution=301,
    min_elevation=80.0,
    ax=axes[1, 1],
    cbar_min=0,
    cbar_max=48.1,
    label="beam_2 & param_2",
)
plt.show()
