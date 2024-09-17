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

# # Predefined instances

from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

import pyant


pprint(pyant.avalible_beams())


e3d_m = pyant.beam_of_radar("e3d_module", "array")
e3d_m_I0 = e3d_m.sph_gain(
    azimuth=e3d_m.azimuth,
    elevation=e3d_m.elevation,
)
print(f"EISCAT 3D Module peak gain: {10*np.log10(e3d_m_I0)}")


e3d = pyant.beam_of_radar("e3d_stage1", "array")
e3d_sparse = pyant.beam_of_radar("e3d_stage1", "array", configuration="sparse")

e3d_I0 = e3d.sph_gain(
    azimuth=e3d.azimuth,
    elevation=e3d.elevation,
)
e3d_sparse_I0 = e3d_sparse.sph_gain(
    azimuth=e3d_sparse.azimuth,
    elevation=e3d_sparse.elevation,
)
print(f"EISCAT 3D Stage 1 peak gain (dense core): {10*np.log10(e3d_I0)}")
print(f"EISCAT 3D Stage 1 peak gain (sparse core): {10*np.log10(e3d_sparse_I0)}")


e3d_s2 = pyant.beam_of_radar("e3d_stage2", "array")

e3d_s2_I0 = e3d_s2.sph_gain(
    azimuth=e3d_s2.azimuth,
    elevation=e3d_s2.elevation,
)
print(f"EISCAT 3D Stage 2 peak gain: {10*np.log10(e3d_s2_I0)}")


pyant.plotting.antenna_configuration(e3d.antennas)
pyant.plotting.antenna_configuration(e3d_sparse.antennas)


pyant.plotting.gain_heatmap(e3d, resolution=100, min_elevation=80.0)
e3d.sph_point(azimuth=45, elevation=30.0)
pyant.plotting.gain_heatmap(e3d, resolution=100, min_elevation=80.0)
plt.show()
