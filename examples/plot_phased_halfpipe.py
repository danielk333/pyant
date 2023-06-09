"""
Phase steerable Half-pipe
============================
"""

import itertools as it

import numpy as np
import matplotlib.pyplot as plt

import pyant


def flatten(Lst):
    return list(it.chain(*Lst))


phi = np.linspace(0, 2 * np.pi, 100)
cosp = np.cos(phi)
sinp = np.sin(phi)
aa = np.cos(np.radians(90 - np.array([75, 60, 45, 30])))

pa = flatten([(a * cosp, a * sinp, "w-") for a in aa])

beam = pyant.models.PhasedFiniteCylindricalParabola(
    azimuth=0,
    elevation=90.0,
    depth=18,
    phase_steering=[0.0, 7.0, 14.0, 21.0],
    frequency=30e6,
    width=120.0,
    height=40.0,
    degrees=True,
)

fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=80)
axes = axes.flatten()
for i in range(beam.named_shape["phase_steering"]):
    pyant.plotting.gain_heatmap(
        beam,
        resolution=901,
        min_elevation=30.0,
        ax=axes[i],
        ind={
            "phase_steering": i,
        },
    )
    axes[i].plot(*pa)
    axes[i].set_title(f"{int(beam.phase_steering[i])} deg steering")

beam_two = pyant.models.PhasedFiniteCylindricalParabola(
    azimuth=0,
    elevation=[90.0, 60.0, 30.0],
    depth=18,
    phase_steering=[0, 25.0],
    frequency=224.0e6,
    width=120.0 / 4,
    height=40.0,
    degrees=True,
)


fig, axes = plt.subplots(2, 3, figsize=(10, 6), dpi=80)
for i in range(beam_two.named_shape["phase_steering"]):
    for j in range(beam_two.named_shape["pointing"]):
        ind = {
            "phase_steering": i,
            "pointing": j,
        }
        pyant.plotting.gain_heatmap(
            beam_two,
            resolution=901,
            min_elevation=20.0,
            ax=axes[i, j],
            ind=ind,
        )

        params, shape = beam_two.get_parameters(ind, named=True)
        phi = params["phase_steering"]
        G0 = beam_two.gain_tf(theta=0, phi=phi, params=params, degrees=True)
        print(
            f"Pointing: az={beam_two.azimuth} deg, el={beam_two.elevation[j]} deg, "
            + f"phase steering={phi} deg "
            + f"-> Peak gain = {np.log10(G0[0])*10:8.3f} dB"
        )

        axes[i, j].plot(*pa)
        axes[i, j].set_title(f"{int(phi)} ph-st | {int(beam_two.elevation[j])} el")

plt.show()
