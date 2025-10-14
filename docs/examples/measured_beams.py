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
phi = np.linspace(0, 8, num=num)
gains = np.exp(-phi) * (np.sin(phi) ** 2 + 0.01)
gains = gains / np.max(gains)

beam = pyant.models.MeasuredAzimuthallySymmetric(
    pointing=np.array([0, 0, 1], dtype=np.float64),
    off_axis_angle=phi,
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
