"""
Antenna array gain
===========================
"""
# import time
import numpy as np
import matplotlib.pyplot as plt

import pyant

np.random.seed(42)

delta_arc = 1.1
samples = 100
range_coef = 5.0

# https://math.stackexchange.com/a/2216736
antennas = np.zeros((3, samples))
for ind in range(1, samples):
    d_theta = delta_arc / np.sqrt(1 + antennas[0, ind - 1] ** 2)
    antennas[0, ind] = antennas[0, ind - 1] + d_theta
    antennas[2, ind] = range_coef * antennas[0, ind]

antennas = pyant.coordinates.sph_to_cart(antennas, degrees=False)
x = antennas[0, :]
y = antennas[1, :]
D = np.sqrt((x[None, :] - x[:, None])**2 + (y[None, :] - y[:, None])**2)

antennas = antennas.reshape((3, 1, samples))

beam = pyant.models.Array(
    azimuth=90,
    elevation=80.0,
    frequency=46.5e6,
    antennas=antennas,
)

M = np.exp(1j*np.pi*2.0*D/beam.wavelength)
diags = np.eye(samples, dtype=bool)
off_diags = np.logical_not(diags)
M[diags] = 1
M[off_diags] = M[off_diags]*(1/D[off_diags])**0.2

beam.mutual_coupling_matrix = M

# k = pyant.coordinates.sph_to_cart(
#     np.array([0, 85, 1], dtype=np.float64),
#     degrees=True,
# )
# x = beam.signals(k, beam.polarization)


# Uncomment these to try the speed up for more complex gain calculations

# start_time = time.time()
# pyant.plotting.gain_heatmap(beam, resolution=100, min_elevation=80.0, vectorized=False)
# print(f'"gain_heatmap" ({100**2}) loop       performance: {time.time() - start_time:.1e} seconds')

# start_time = time.time()
# pyant.plotting.gain_heatmap(beam, resolution=100, min_elevation=80.0, vectorized=True)
# print(f'"gain_heatmap" ({100**2}) vectorized performance: {time.time() - start_time:.1e} seconds')

fig = plt.figure()
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
mat_ax[1].matshow(np.angle(M))

pyant.plotting.gain_heatmap(
    beam,
    resolution=100,
    min_elevation=80.0,
    centered=False,
    ax=axes[1],
)
axes[1].set_title('With MCM')

beam.mutual_coupling_matrix = None

pyant.plotting.gain_heatmap(
    beam,
    resolution=100,
    min_elevation=80.0,
    centered=False,
    ax=axes[2],
)
axes[2].set_title('Without MCM')

pyant.plotting.show()
