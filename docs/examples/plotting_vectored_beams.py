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

# # Vectored parameters

import matplotlib.pyplot as plt
import numpy as np
import pyant


beam = pyant.models.Airy(
    azimuth=[0, 45.0, 0],
    elevation=[90.0, 80.0, 60.0],
    frequency=[930e6, 230e6],
    I0=10**4.81,
    radius=23.0,
    degrees=True,
)
k = np.array([[0, 0, 1.0], [0, 1, 1]]).T


fig, axes = plt.subplots(3, 2, figsize=(10, 6), dpi=80)
for i in range(beam.shape[0]):
    for j in range(beam.shape[1]):
        pyant.plotting.gain_heatmap(
            beam,
            resolution=301,
            min_elevation=80.0,
            ax=axes[i, j],
            ind={
                "pointing": i,
                "frequency": j,
            },
        )
        pstr = f"az={beam.azimuth[i]:.1f} deg | el={beam.elevation[i]:.1f} deg"
        axes[i, j].set_title(f"{pstr} | f:{beam.frequency[j]/1e6:.1f} MHz")
plt.show()
