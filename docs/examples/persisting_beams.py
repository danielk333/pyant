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

import pathlib
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import pyant

# # Persisting beam patterns

beam = pyant.models.Airy(
    pointing=np.array([0, 0, 1], dtype=np.float64),
    frequency=930e6,
    radius=23.0,
    peak_gain=10**4.81,
)
with tempfile.TemporaryDirectory() as tmp:
    path = pathlib.Path(tmp) / "my_beam.npz"
    beam.to_npz(path)
    beam_two = pyant.models.Airy.from_npz(path)

# +
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
pyant.plotting.gain_heatmap(beam, resolution=301, min_elevation=85.0, ax=ax1)
ax1.set_title("Airy")

pyant.plotting.gain_heatmap(beam_two, resolution=301, min_elevation=85.0, ax=ax2)
ax2.set_title("Airy loaded form file")
plt.show()
