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

# # Gaussian planar array gain

import numpy as np
import matplotlib.pyplot as plt
import spacecoords.spherical as sph
import pyant

# +
# # The basic concepts
#
# Assume we have an appropriate coordinate system for our use, most often this is the horizontal
# local coordinate system where zenith is the normal vector to the given location on a sphereroid
# such as WGS84, with the x axis aligned towards East and the y axis aligned towards the North, the
# so called ENU system.
#
# Given a radar with a circular aperture located on a flat plane with normal vector $\mathbf{n}$,
# the resulting gain model can be described, using Fourier Optics, as a 2D Fourier transform of that
# aperture but with the coordinate system rotated so that the normal vector aligns with Z.
#
# Furthermore, assume this radar aperture is actually a phased antenna array with a given pointing
# $\mathbf{p}$.


normal = np.array([0.0, 0.0, 1.0])
pointing = sph.sph_to_cart(np.array([0.0, 45.0, 1.0]), degrees=True)


# +
# The first important coordinate system being used is the one spanned by the Normal and two
# orthogonal vectors inside the aperture itself. We can define these by taking cross products
# between the pointing and the normal vectors (in the actual code, the special case of
# parallel pointing and normal is handled)
#
# We can call these ${p, ct, ht}$ where ct is in the azimuthal direction compared to the
# pointing and ht is in the direction of the pointing.

ct = np.cross(pointing, normal)
ct = ct / np.linalg.norm(ct, axis=0)

ht = np.linalg.cross(normal, ct, axis=0)
ht = ht / np.linalg.norm(ht, axis=0)

# +
# The second important coordinate system is the one aligned with the z axis towards with the
# pointing direction itself and otherwise aligned towards the previous system.
# We can call the basis of this system {n, ct, ot} where ct is as defined before and ot is
# simply the orthonormal completion of the basis which produces a vector aligned towards the
# azimuthal direction of the pointing and perpendicular to that pointing direction

ot = np.cross(pointing, ct, axis=0)
ot = ot / np.linalg.norm(ot, axis=0)

# +
# We can plot this system as this:

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot([0, pointing[0]], [0, pointing[1]], [0, pointing[2]], ls="-", c="b", label="pointing")
ax.plot([0, normal[0]], [0, normal[1]], [0, normal[2]], ls="-", c="k", label="normal")
ax.plot([0, ct[0]], [0, ct[1]], [0, ct[2]], ls="-", c="r", label="ct=p X n")
ax.plot([0, ht[0]], [0, ht[1]], [0, ht[2]], ls="--", c="k", label="ht=n X ct")
ax.plot([0, ot[0]], [0, ot[1]], [0, ot[2]], ls="--", c="b", label="ot=p X ct")
ax.axis("equal")
ax.legend()

# +
#
beam = pyant.models.Gaussian(
    peak_gain=10**4.81,
)
param = pyant.models.GaussianParams(
    pointing=np.array([0, 0, 1], dtype=np.float64),
    normal_pointing=np.array([0, 0, 1], dtype=np.float64),
    frequency=46.5e6,
    radius=100.0,
    beam_width_scaling=1,
)

# +
# # Plot the gain pattern in zenith for two different planar tilts
fig, (ax1, ax2) = plt.subplots(1, 2)
pyant.plotting.gain_heatmap(
    beam,
    param,
    resolution=301,
    min_elevation=80.0,
    ax=ax1,
    cbar_min=0,
    cbar_max=np.log10(beam.peak_gain) * 10,
)
ax1.set_title("Plane normal-elevation = 90 deg")

off_axis = 50
param.normal_pointing = sph.sph_to_cart(np.array([0.0, 90.0 - off_axis, 1.0]), degrees=True)

pyant.plotting.gain_heatmap(
    beam,
    param,
    resolution=301,
    min_elevation=80.0,
    ax=ax2,
    cbar_min=0,
    cbar_max=np.log10(beam.peak_gain) * 10,
)
ax2.set_title(f"Plane normal-elevation = {off_axis} deg")


# +
# # Plot the gain 1 degree off bore-sight as a function of frequency and radius
# This is an example of setting up a larger vectorized calculation using `pyant`
# and then evaluating that calculation and formatting it back to its intended
# shape and parameter space

num = 100
tnum = num**2
freq = np.linspace(50e6, 900e6, num)
radius = np.linspace(23, 100, num)
fmat, rmat = np.meshgrid(freq, radius)
p = np.zeros((3, tnum), dtype=np.float64)
p[2, :] = 1

param_vec = pyant.models.GaussianParams(
    pointing=p,
    normal_pointing=p,
    frequency=fmat.reshape((tnum,)),
    radius=rmat.reshape((tnum,)),
    beam_width_scaling=np.ones((tnum,), dtype=np.float64),
)

k_in = sph.sph_to_cart(np.array([0, 89.0, 1]), degrees=True)

g = beam.gain(k_in, param_vec)

fig, ax = plt.subplots()
pm = ax.pcolormesh(fmat * 1e-6, rmat, 10 * np.log10(g.reshape(fmat.shape)))  # type: ignore
ax.set_xlabel("Frequency [MHz]")
ax.set_ylabel("Radius [m]")
ax.set_title("1-degree off bore-sight gain")
cb = fig.colorbar(pm, ax=ax)
cb.set_label("Gain [dB]")

plt.show()
