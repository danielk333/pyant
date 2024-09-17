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
import pyant


beam = pyant.beam_of_radar("eiscat_uhf", "measured")

# +
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

pyant.plotting.gain_heatmap(beam, min_elevation=80, ax=ax1)
pyant.plotting.gains(beam, min_elevation=80, ax=ax2)

plt.show()
