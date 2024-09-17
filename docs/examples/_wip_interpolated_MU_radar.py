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

# # Interpolated MU radar
#
# Combination of interpolated base antenna field and a measured antenna gain pattern

import pyant

beam = pyant.beam_of_radar(
    "mu",
    "interpolated_array",
    cache_path,
    resolution=(1000, 1000, None),
)
base_beam = pyant.beam_of_radar("mu", "array")
fig, axes = plt.subplots(1, 2)
pyant.plotting.gain_heatmap(
    base_beam,
    resolution=100,
    min_elevation=80.0,
    centered=False,
    ax=axes[0],
)
axes[0].set_title("Array")
pyant.plotting.gain_heatmap(
    beam,
    resolution=100,
    min_elevation=80.0,
    centered=False,
    ax=axes[1],
)
axes[1].set_title("Interpolated array")
plt.show()
