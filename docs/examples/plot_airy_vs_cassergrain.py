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


import matplotlib.pyplot as plt
import pyant


beam = pyant.models.Airy(
    azimuth=0,
    elevation=90.0,
    frequency=930e6,
    I0=10**4.81,
    radius=23.0,
    degrees=True,
)
beam_c = pyant.models.Cassegrain(
    azimuth=0,
    elevation=90.0,
    frequency=930e6,
    I0=10**4.81,
    outer_radius=40.0,
    inner_radius=23.0,
    degrees=True,
)

# +
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
pyant.plotting.gain_heatmap(beam, resolution=301, min_elevation=85.0, ax=ax1)
ax1.set_title("Airy")

pyant.plotting.gain_heatmap(beam_c, resolution=301, min_elevation=85.0, ax=ax2)
ax2.set_title("Cassegrain")
plt.show()
