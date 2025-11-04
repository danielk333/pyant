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

# # Complex response of array

import numpy as np
import matplotlib.pyplot as plt

import pyant

ant_n = 55
dr = 2.0

xv, yv = np.meshgrid(
    np.arange(-ant_n // 2, ant_n // 2) * dr,
    np.arange(-ant_n // 2, ant_n // 2) * dr,
)
antennas = np.zeros((3, ant_n**2, 1))
antennas[0, :, 0] = xv.flatten()
antennas[1, :, 0] = yv.flatten()

beam = pyant.models.Array(
    antennas=antennas,
)
param = pyant.models.ArrayParams(
    pointing=np.array([0, 0, 1], dtype=np.float64),
    frequency=46.5e6,
    polarization=np.array([1, 1j]) / np.sqrt(2),
)
param_lin = param.copy()
param_lin.polarization = np.array([1.0, 0.0])

k = np.array([0, 0, 1])
km = np.array([[0, 0, 1], [0, 0.1, 0.9], [0, 0.1, 0.8]]).T

print(f"Gain LHCP: {beam.gain(k, param)}")
print(f"Gain LHCP: {beam.gain(km, param)}")

print(f"Gain |H>: {beam.gain(k, param_lin)}")
print(f"Gain |H>: {beam.gain(km, param_lin)}")

print(
    f"\nsignals response from {k} of LHCP analysis when signal is |H>:\n"
    f"{beam.signals(k, param_lin)}"
)
print(
    f"signals response from {k} of LHCP analysis when signal is LHCP:\n"
    f"{beam.signals(k, param)}\n"
)

beam.polarization = np.array([1.0, 0.0])
print(
    f"\nGain from {k} of |H> analysis when signal is |H>:\n"
    f"{beam.gain(k, param_lin)}"
)
print(
    f"Gain from {k} of |H> analysis when signal is LHCP:\n"
    f"{beam.gain(k, param)}\n"
)
beam.polarization = np.array([1, 1j]) / np.sqrt(2)

fig, (ax1, ax2) = plt.subplots(1, 2)
pyant.plotting.gain_heatmap(beam, param, resolution=200, min_elevation=60.0, ax=ax1)
pyant.plotting.antenna_configuration(beam.antennas, z_axis=False, ax=ax2)

fig, ax = plt.subplots()
pyant.plotting.polarization_heatmap(
    beam, param, k, resolution=200, ax=ax
)
plt.show()
