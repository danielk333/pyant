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

# # Antenna array gain


# import time
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import pyant


antenna_num = 100
beam = pyant.beams.arrays.equidistant_archimedian_spiral(
    antenna_num=antenna_num,
    arc_separation=1.1,
    range_coefficient=5.0,
)
param = pyant.models.ArrayParams(
    frequency=46.5e6,
    pointing=np.array([0, 0, 1], dtype=np.float64),
    polarization=beam.polarization.copy(),
)
wavelength = const.c / param.frequency

x = beam.antennas[0, 0, :]
y = beam.antennas[1, 0, :]
d = np.sqrt((x[None, :] - x[:, None]) ** 2 + (y[None, :] - y[:, None]) ** 2)

m = np.exp(1j * np.pi * 2.0 * d / wavelength)
diags = np.eye(antenna_num, dtype=bool)
off_diags = np.logical_not(diags)
m[diags] = 1
m[off_diags] = m[off_diags] * (1 / d[off_diags]) ** 0.2

beam.mutual_coupling_matrix = m


# +
fig = plt.figure(figsize=(8, 8))
axes = [
    fig.add_subplot(2, 2, 1, projection="3d"),
    fig.add_subplot(2, 2, 2),
    fig.add_subplot(2, 2, 4),
]
mat_ax = [
    fig.add_subplot(2, 3, 4),
    fig.add_subplot(2, 3, 5),
]

pyant.plotting.antenna_configuration(beam.antennas, ax=axes[0])
mat_ax[0].matshow(np.abs(m))
mat_ax[0].set_title("MCM magnitude")
mat_ax[0].set_xlabel("Rows")
mat_ax[0].set_ylabel("Columns")

mat_ax[1].matshow(np.angle(m))
mat_ax[1].set_title("MCM angle")
mat_ax[1].set_xlabel("Rows")

pyant.plotting.gain_heatmap(
    beam,
    param,
    resolution=100,
    min_elevation=80.0,
    centered=False,
    ax=axes[1],
)
axes[1].set_title("With MCM")

beam.mutual_coupling_matrix = None

pyant.plotting.gain_heatmap(
    beam,
    param,
    resolution=100,
    min_elevation=80.0,
    centered=False,
    ax=axes[2],
)
axes[2].set_title("Without MCM")

plt.show()
