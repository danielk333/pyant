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
import matplotlib.pyplot as plt
import pyant


antenna_num = 100
beam = pyant.beams.arrays.equidistant_archimedian_spiral(
    antenna_num=antenna_num,
    arc_separation=1.1,
    range_coefficient=5.0,
    frequency=46.5e6,
)

x = beam.antennas[0, 0, :]
y = beam.antennas[0, 1, :]
D = np.sqrt((x[None, :] - x[:, None]) ** 2 + (y[None, :] - y[:, None]) ** 2)

M = np.exp(1j * np.pi * 2.0 * D / beam.wavelength)
diags = np.eye(antenna_num, dtype=bool)
off_diags = np.logical_not(diags)
M[diags] = 1
M[off_diags] = M[off_diags] * (1 / D[off_diags]) ** 0.2

beam.mutual_coupling_matrix = M


# Uncomment these to try the speed up for more complex gain calculations

# start_time = time.time()
# pyant.plotting.gain_heatmap(beam, resolution=100, min_elevation=80.0, vectorized=False)
# print(f'"gain_heatmap" ({100**2}) loop       performance: {time.time() - start_time:.1e} seconds')

# start_time = time.time()
# pyant.plotting.gain_heatmap(beam, resolution=100, min_elevation=80.0, vectorized=True)
# print(f'"gain_heatmap" ({100**2}) vectorized performance: {time.time() - start_time:.1e} seconds')


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
mat_ax[0].matshow(np.abs(M))
mat_ax[0].set_title("MCM magnitude")
mat_ax[0].set_xlabel("Rows")
mat_ax[0].set_ylabel("Columns")

mat_ax[1].matshow(np.angle(M))
mat_ax[1].set_title("MCM angle")
mat_ax[1].set_xlabel("Rows")

pyant.plotting.gain_heatmap(
    beam,
    resolution=100,
    min_elevation=80.0,
    centered=False,
    ax=axes[1],
)
axes[1].set_title("With MCM")

beam.mutual_coupling_matrix = None

pyant.plotting.gain_heatmap(
    beam,
    resolution=100,
    min_elevation=80.0,
    centered=False,
    ax=axes[2],
)
axes[2].set_title("Without MCM")

plt.show()
