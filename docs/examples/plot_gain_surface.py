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

import numpy as np
import matplotlib.pyplot as plt
import pyant


beam = pyant.models.Cassegrain(
    pointing=np.array([0, 0, 1], dtype=np.float64),
    frequency=930e6,
    peak_gain=10**4.81,
    outer_radius=40.0,
    inner_radius=23.0,
)


# +
pyant.plotting.gain_surface(beam, resolution=301, min_elevation=85.0)
plt.show()
