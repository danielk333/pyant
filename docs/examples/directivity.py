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

# # Aperture to gain

import numpy as np
import scipy.constants
import matplotlib.pyplot as plt
import spacecoords.linalg as linalg
import spacecoords.spherical as sph
import pyant


frequency = 46.5e6
wavelength = scipy.constants.c / frequency
width = 103
beam_arr = pyant.beams.circular_array(array_radius=width / 2, antenna_spacing=0.65 * wavelength)
param_arr = pyant.models.ArrayParams(
    frequency=frequency,
    pointing=np.array([0, 0, 1], dtype=np.float64),
    polarization=beam_arr.polarization.copy(),
)

num = 100_000
ant_eff = 0.8
int_g = pyant.statistics.monte_carlo_integrate_gain_hemisphere(beam_arr, param_arr, num)
print(f"{int_g=}")
beam_arr.scaling_factor = ant_eff * np.pi * 4 / int_g
print(f"{beam_arr.scaling_factor=}")

print(f"{beam_arr.antenna_number=}")

peak_gain_dB = np.log10(beam_arr.gain(param_arr.pointing, param_arr) / np.sqrt(2)) * 10
print(f"{peak_gain_dB=}")

num = 1000
zang = np.linspace(0, 10, num=num)
k_vecs = sph.az_el_point(azimuth=np.zeros((num,)), elevation=90 - zang, degrees=True)
g = beam_arr.gain(k_vecs, param_arr) / np.sqrt(2)

beam_width = zang[np.argmax(g < g.max() / 2)] * 2
print(f"{beam_width=} deg")
fig, ax = plt.subplots()
ax.plot(zang, np.log10(g) * 10)
ax.axvline(beam_width / 2)
plt.show()
exit()

fig, ax = plt.subplots(1, figsize=(7, 7))

pyant.plotting.gain_heatmap(
    beam_arr,
    param_arr,
    resolution=301,
    min_elevation=80.0,
    ax=ax,
    cbar_min=0,
)
ax.set_title("Circular antenna array")

plt.show()
