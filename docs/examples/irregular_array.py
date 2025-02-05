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

# # Irregular array


# import time
import matplotlib.pyplot as plt
import pyant


antenna_num = 100
beam = pyant.beam_of_radar("pansy", "array")

print(beam.meta)

# +
pyant.plotting.antenna_configuration(beam.antennas)


# +
fig, ax = plt.subplots()
pyant.plotting.gain_heatmap(
    beam,
    resolution=300,
    min_elevation=80.0,
    centered=False,
    ax=ax,
)
ax.set_title("PANSY radar")

plt.show()
