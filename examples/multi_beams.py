"""
Collection of beams
===================

This assumes that the signals in incoherently integrated between beams,
i.e. gain in linearly additive.
"""
import pyant
import numpy as np
import matplotlib.pyplot as plt


beams = [
    pyant.models.FiniteCylindricalParabola(
        azimuth=0,
        elevation=el,
        frequency=224.0e6,
        I0=10**4.81,
        width=30.0,
        height=40.0,
        degrees=True,
    )
    for el in [90.0, 80.0, 70.0, 60.0]
]

k = np.array([0, 0, 1])

gains = [b.gain(k) for b in beams]
print(f"Individual gains {np.log10(gains)*10} dB")

gain_sum = np.sum(gains)
print(f"Summed gains {np.log10(gain_sum)*10} dB")

print(f"Gain of beam 2 {beams[1].gain(k)}")

fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=80)
for beam, ax in zip(beams, axes.flatten()):
    pyant.plotting.gain_heatmap(beam, min_elevation=60, ax=ax)


summed_beam = pyant.SummedBeams(beams)

fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
pyant.plotting.gain_heatmap(summed_beam, min_elevation=0, ax=ax)

plt.show()
