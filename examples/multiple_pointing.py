"""
Multiple pointing directions
=============================
"""
import matplotlib.pyplot as plt
import numpy as np
import pyant

beam = pyant.models.Airy(
    azimuth=[0, 45.0, 0],
    elevation=[90.0, 80.0, 60.0],
    frequency=[930e6, 230e6],
    I0=10**4.81,
    radius=23.0,
)

print(f"Gain can be calculated with the parameter sizes: {beam.shape}")
print(f"Corresponding to the following parameters: {beam.keys}")
print("These are the default values:")

for key, val in beam.get_parameters(named=True).items():
    print(f"{key}: {val}")

k = np.array([[0, 0, 1.0], [0, 1, 1]]).T
G = beam.gain(k[:, 0])

print(f"Gain for {k[:,0]} without giving parameter index: {G}")

for pi in range(len(beam.azimuth)):
    print(
        f'Gain for {k[:,0]} and pointing {pi} (freq = 0 by default): \
        {beam.gain(k[:,0], ind={"pointing":pi})}'
    )
print(f'Gain for {k} and freq = 1: {beam.gain(k, ind={"frequency":1})}')

fig, axes = plt.subplots(3, 2, figsize=(10, 6), dpi=80)
for i in range(beam.shape[0]):
    for j in range(beam.shape[1]):
        pyant.plotting.gain_heatmap(
            beam,
            resolution=301,
            min_elevation=80.0,
            ax=axes[i, j],
            ind={
                "pointing": i,
                "frequency": j,
            },
        )
        pstr = " ".join([f"{x:.2e}" for x in beam.pointing[:, i]])
        axes[i, j].set_title(f"p: {pstr} | f:{beam.frequency[j]}")

pyant.plotting.show()
