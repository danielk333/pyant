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


# # Measured beam pattern


import matplotlib.pyplot as plt
import numpy as np
import pyant

num = 15
els = np.linspace(82, 90, num=num)
gains = np.exp(-(90 - els)) * (np.sin(els) ** 2 + 0.01)
gains = gains / np.max(gains)

beam = pyant.models.MeasuredAzimuthallySymmetric(
    elevations=els,
    gains=gains,
    interpolation_method="linear",
    degrees=True,
)

# +
fig, axes = plt.subplots(2, 2, figsize=(12, 5))

pyant.plotting.gain_heatmap(beam, min_elevation=80, ax=axes[0, 0])
pyant.plotting.gains(beam, min_elevation=80, ax=axes[0, 1])
for ax in axes[0, :]:
    ax.set_title("Gain pattern - linear interpolation")

beam.interpolation_method = "cubic_spline"

pyant.plotting.gain_heatmap(beam, min_elevation=80, ax=axes[1, 0])
pyant.plotting.gains(beam, min_elevation=80, ax=axes[1, 1])
for ax in axes[1, :]:
    ax.set_title("Gain pattern - clubic_spline interpolation")

fig.tight_layout()
plt.show()
