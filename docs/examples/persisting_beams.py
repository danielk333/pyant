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

import pathlib
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import spacecoords.spherical as sph
import pyant

# # Persisting beam patterns

k = sph.az_el_point(azimuth=0, elevation=89, degrees=True)
num = 500
fs = np.linspace(200e6, 930e6, num=num)
beam = pyant.models.Airy(
    peak_gain=10**4.81,
)
param = pyant.models.AiryParams(
    pointing=np.array([0, 0, 1], dtype=np.float64),
    frequency=930e6,
    radius=23.0,
)
vector_param = pyant.models.AiryParams.replace_and_broadcast(
    parameters=param,
    new_parameters=dict(frequency=fs),
)
with tempfile.TemporaryDirectory() as tmp:
    path = pathlib.Path(tmp) / "my_params.npz"
    param.to_npz(path)
    param_two = pyant.models.AiryParams.from_npz(path)

    path = pathlib.Path(tmp) / "my_beam.json"
    beam.to_json(path)
    beam_two = pyant.models.Airy.from_json(path)

    path = pathlib.Path(tmp) / "my_vec_params.npz"
    vector_param.to_npz(path)
    vector_param_two = pyant.models.AiryParams.from_npz(path)


# +
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
pyant.plotting.gain_heatmap(beam, param, resolution=301, min_elevation=85.0, ax=ax1)
ax1.set_title("Airy")

pyant.plotting.gain_heatmap(beam_two, param_two, resolution=301, min_elevation=85.0, ax=ax2)
ax2.set_title("Airy loaded form file")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(fs*1e-6, 10*np.log10(beam.gain(k, vector_param)))
ax1.set_xlabel("Frequency [MHz]")
ax1.set_ylabel("Gain @ 89 elevation [dB]")
ax1.set_title("Airy")

ax2.plot(fs*1e-6, 10*np.log10(beam_two.gain(k, vector_param_two)))
ax2.set_xlabel("Frequency [MHz]")
ax2.set_ylabel("Gain @ 89 elevation [dB]")
ax2.set_title("Airy loaded form file")

plt.show()
