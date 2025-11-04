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

# # Phase steerable Half-pipe

import itertools as it
import numpy as np
import matplotlib.pyplot as plt
import spacecoords.spherical as sph
import pyant


def flatten(Lst):
    return list(it.chain(*Lst))


phi = np.linspace(0, 2 * np.pi, 100)
cosp = np.cos(phi)
sinp = np.sin(phi)
phase_st = np.array([0, 20, 40, 60], dtype=np.float64)
aa = np.cos(np.radians(90 - phase_st))

pa = flatten([(a * cosp, a * sinp, "w-") for a in aa])

beam = pyant.models.PhasedFiniteCylindricalParabola()
param = pyant.models.PhasedFiniteCylindricalParabolaParams(
    pointing=np.array([0, 0, 1], dtype=np.float64),
    phase_steering=0,
    frequency=30e6,
    width=120.0,
    height=40.0,
    depth=18.0,
    aperture_width=120.0,
)


# +
fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=80)
axes = axes.flatten()
for i in range(len(phase_st)):
    param.phase_steering = beam._nominal_phase_steering(phase_st[i], degrees=True)
    pyant.plotting.gain_heatmap(
        beam,
        param,
        resolution=901,
        min_elevation=0.0,
        cbar_min=0,
        ax=axes[i],
    )
    axes[i].plot(*pa)
    axes[i].set_title(f"{int(phase_st[i])} deg steering")

elevation = [90.0, 60.0, 30.0]
phase_steering = [0, 25.0]

param_two = pyant.models.PhasedFiniteCylindricalParabolaParams(
    pointing=np.array([0, 0, 1], dtype=np.float64),
    phase_steering=0,
    frequency=224.0e6,
    width=120.0 / 4,
    height=40.0,
    depth=18.0,
    aperture_width=120.0,
)


fig, axes = plt.subplots(2, 3, figsize=(10, 6), dpi=80)
for i in range(len(phase_steering)):
    for j in range(len(elevation)):
        param_two.pointing = sph.az_el_point(azimuth=0, elevation=elevation[j], degrees=True)
        param_two.phase_steering = beam._nominal_phase_steering(phase_steering[i], degrees=True)
        pyant.plotting.gain_heatmap(
            beam,
            param_two,
            resolution=901,
            min_elevation=20.0,
            ax=axes[i, j],
            cbar_min=0,
        )

        g0 = beam.gain_tf(theta=0, phi=np.radians(phase_steering[i]), parameters=param_two)
        print(
            f"Pointing: az=0 deg, el={elevation[j]} deg, "
            + f"phase steering={phase_steering[i]} deg "
            + f"-> Peak gain = {np.log10(g0)*10:8.3f} dB"
        )

        axes[i, j].plot(*pa)
        axes[i, j].set_title(f"{int(phase_steering[i])} ph-st | {int(elevation[j])} el")


plt.show()
