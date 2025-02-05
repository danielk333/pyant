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
    azimuth=0,
    elevation=90.0,
    frequency=46.5e6,
    antennas=antennas,
    degrees=True,
)

beam_linp = pyant.models.Array(
    azimuth=0,
    elevation=90.0,
    frequency=46.5e6,
    polarization=np.array([1, 0]),
    antennas=antennas,
    degrees=True,
)

k = np.array([0, 0, 1])
km = np.array([[0, 0, 1], [0, 0.1, 0.9], [0, 0.1, 0.8]]).T


print(f"Gain LHCP: {beam.gain(k)}")
print(f"Gain LHCP: {beam.gain(km)}")

print(f"Gain |H>: {beam_linp.gain(k)}")
print(f"Gain |H>: {beam_linp.gain(km)}")

print(
    f"\nsignals response from {k} of LHCP analysis when signal is |H>:\n"
    f"{beam.signals(k=k, polarization=np.array([1, 0]))}"
)
print(
    f"signals response from {k} of LHCP analysis when signal is LHCP:\n"
    f"{beam.signals(k=k, polarization=beam.polarization)}\n"
)

print(
    f"\nGain from {k} of LHCP analysis when signal is |H>:\n"
    f"{beam.gain(k=k, polarization=np.array([1, 0]))}"
)
print(
    f"Gain from {k} of LHCP analysis when signal is LHCP:\n"
    f"{beam.gain(k=k, polarization=beam.polarization)}\n"
)

print(
    f"\nGain from {k} of |H> analysis when signal is |H>:\n"
    f"{beam_linp.gain(k=k, polarization=np.array([1, 0]))}"
)
print(
    f"Gain from {k} of |H> analysis when signal is LHCP:\n"
    f"{beam_linp.gain(k=k, polarization=beam.polarization)}\n"
)
