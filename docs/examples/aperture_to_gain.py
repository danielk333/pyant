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

# +
# # The basics
# TODO: this needs to be connected to more concrete theory
#
# To gain an understanding about the most commonly used formulas for the gain pattern of different
# radars we need to start by looking at a typical case and build up from there.
#
# So lets start with a sketch of a parabolic receiver dish antenna with a un-obstructing
# radiating antenna at the focus, which radiates uniformly the entire reflector

num = 1000
line_num = 10
focus_height = 5
width = 32
start_ray = 50
freq = 200e6
wavelength = scipy.constants.c / freq


def parabolic_func(x):
    return 1 / (4 * focus_height) * x**2


def dx_parabolic_func(x):
    return 1 / (2 * focus_height) * x


y_max = parabolic_func(-width / 2)


def ray(x0, res=200, ray_angle=0):
    y0 = parabolic_func(x0)
    p0 = np.array([x0, y0])

    tangent_vec = np.array([1, dx_parabolic_func(x0)])
    normal_vec = linalg.rot_mat_2d(np.pi / 2) @ tangent_vec
    ray_vec_back = linalg.rot_mat_2d(-ray_angle) @ np.array([0, 1])

    ang = np.arccos(np.dot(normal_vec / np.linalg.norm(normal_vec), ray_vec_back))

    new_ray_vec = linalg.rot_mat_2d(np.sign(x0) * 2 * ang) @ ray_vec_back

    t = np.linspace(0, y0 + focus_height, res)
    t_back = np.linspace(0, start_ray - y0, res)

    ps = p0[:, None] + new_ray_vec[:, None] * t[None, :]
    ps_top = p0[:, None] + ray_vec_back[:, None] * t_back[None, :]

    # incident angle
    xs = np.concatenate([ps_top[0, :], ps[0, :]])
    ys = np.concatenate([ps_top[1, :], ps[1, :]])
    return xs, ys


xs = np.linspace(-width / 2, width / 2, num=num)
ys = parabolic_func(xs)
ap = np.full_like(xs, y_max)

fig, ax = plt.subplots()

ax.plot(xs, ys, ls="-", c="k", label="reflector")
ax.plot(xs, ap, ls="--", c="b", label="aperture")
ax.plot([0], [focus_height], "ok", label="focus")

ray_xs = np.linspace(-width / 2 * 0.9, width / 2 * 0.9, num=line_num)
for ind in range(len(ray_xs)):
    rxs, rys = ray(ray_xs[ind])
    ax.plot(rxs, rys, ls="-", c="r", alpha=0.5)

ax.legend()
ax.axis("equal")
plt.show()


# +
# Because of the definition of a parabola where all distances between the focus, the reflecting
# surface, and the input aperture are the same by definition, we can assume the wave has the same
# phase and illumination across the aperture.
#
# If we here apply the Huygens–Fresnel principle at each point of the aperture, one way of
# visualizing it is thinking about it as a slit which the already plane wave (made plane by the
# reflector) passes trough. It can be shown that the equations for the far field radiation
# pattern in this case is exactly the two dimensional Fourier transform of the electromagnetic
# field over the aperture itself.
#
# In this way, a radar dish can be thought of as a planar aperture, and the radiation pattern easily
# calculated using a Fourier transform. However, this does not only hold for transmission but also
# for reception because we can just reverse the wave-directions in the argumentation and arrive at
# the same pattern.
#
# `pyant` has this pattern implemented: the Airy disk. Lets see how this looks like as a line plot
# of gain as a function of incoming angle

zenith_angle = np.linspace(0, 10, num=500)

beam = pyant.models.Airy()
param = pyant.models.AiryParams(
    pointing=np.array([0, 0, 1], dtype=np.float64),
    frequency=freq,
    radius=width / 2,
)
g = beam.sph_gain(
    azimuth=np.zeros_like(zenith_angle),
    elevation=90 - zenith_angle,
    parameters=param,
    degrees=True,
)

fig, ax = plt.subplots()
ax.plot(zenith_angle, 10 * np.log10(g), label="Airy gain")
ax.set_xlabel("Zenith angle [deg]")
ax.set_ylabel("Gain [dB]")
ax.legend()
plt.show()

# +
# Of course we could also apply more fundamental equations at the surface of the reflector and solve
# the full resulting electromagnetic wave, however this would require much more complicated
# equations and our current method is for now sufficient.
#
# For planar antenna arrays, the Huygens–Fresnel principle also work because here "each" point on
# the "aperture" is actually a small antenna radiating a wave modified by the individual antennas
# radiation pattern (like the inclination factor in the classical Huygens–Fresnel formulation).
#
# For antenna arrays we can usually calculate what we call the "array factor" which is basically
# just the sum over all antennas receiving a plane wave with some incoming wave direction. Let us
# examine the difference between the array factor and the Fourier transform of the aperture.
#
# We will see very similar main lobes, however, the side lobes are significantly different. Also,
# this comparison breaks down when we consider pointing the phased array off-axis as we will see in
# the next section.

beam_arr = pyant.beams.circular_array(array_radius=width / 2, antenna_spacing=4 * wavelength / 2)
param_arr = pyant.models.ArrayParams(
    frequency=freq,
    pointing=np.array([0, 0, 1], dtype=np.float64),
    polarization=beam_arr.polarization.copy(),
)
beam.peak_gain = np.sqrt(2) * beam_arr.antenna_number

fig, ax = plt.subplots(1, figsize=(7, 7))
pyant.plotting.antenna_configuration(beam_arr.antennas, z_axis=False, ax=ax)
ax.axis("equal")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
pyant.plotting.gain_heatmap(
    beam,
    param,
    resolution=301,
    min_elevation=80.0,
    ax=ax1,
    cbar_min=0,
)
ax1.set_title("Airy disk")

pyant.plotting.gain_heatmap(
    beam_arr,
    param_arr,
    resolution=301,
    min_elevation=80.0,
    ax=ax2,
    cbar_min=0,
)
ax2.set_title("Circular antenna array")

plt.show()

# +
# If we just aim to model the main lobe, we could simply assume that the inclination factor and the
# illumination factor are simple Gaussian functions. Then, the 2D Fourier transform becomes that of
# a Gaussian with different widths in different axis relative the pointing direction.

beam_g = pyant.models.Gaussian(peak_gain=beam.peak_gain)
param_g = pyant.models.GaussianParams(
    pointing=np.array([0, 0, 1], dtype=np.float64),
    normal_pointing=np.array([0, 0, 1], dtype=np.float64),
    frequency=freq,
    radius=width / 2,
    beam_width_scaling=1,
)
param_g_scaling = pyant.models.GaussianParams(
    pointing=np.array([0, 0, 1], dtype=np.float64),
    normal_pointing=np.array([0, 0, 1], dtype=np.float64),
    frequency=freq,
    radius=width / 2,
    beam_width_scaling=1.5,
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
pyant.plotting.gain_heatmap(
    beam,
    param,
    resolution=301,
    min_elevation=80.0,
    ax=ax1,
    cbar_min=0,
)
ax1.set_title("Airy disk")

pyant.plotting.gain_heatmap(
    beam_g,
    param_g,
    resolution=301,
    min_elevation=80.0,
    ax=ax2,
    cbar_min=0,
)
ax2.set_title("Gaussian taperd array")

fig, ax = plt.subplots(figsize=(12, 5))
ax.axhline(
    np.log10(beam.peak_gain / 2) * 10.0,
    ls="--",
    c="r",
    label="-3 dB point",
)
pyant.plotting.gains(
    [beam, beam_arr, beam_g, beam_g],
    [param, param_arr, param_g, param_g_scaling],
    min_elevation=80,
    legends=[
        "Airy",
        "Array",
        "Gaussian",
        f"Gaussian (width scaling={param_g_scaling.beam_width_scaling})",
    ],
    ax=ax,
)
ax.set_ylim(0, None)

off_axis = 30
param.pointing = sph.sph_to_cart(np.array([0.0, 90.0 - off_axis, 1.0]), degrees=True)
param_arr.pointing = sph.sph_to_cart(np.array([0.0, 90.0 - off_axis, 1.0]), degrees=True)
param_g.pointing = sph.sph_to_cart(np.array([0.0, 90.0 - off_axis, 1.0]), degrees=True)
param_g_scaling.pointing = sph.sph_to_cart(np.array([0.0, 90.0 - off_axis, 1.0]), degrees=True)


fig, ax = plt.subplots(figsize=(12, 5))
ax.axhline(
    np.log10(beam.peak_gain / 2) * 10.0,
    ls="--",
    c="r",
    label="-3 dB point",
)
pyant.plotting.gains(
    [beam, beam_arr, beam_g, beam_g],
    [param, param_arr, param_g, param_g_scaling],
    min_elevation=95 - off_axis,
    max_elevation=85 - off_axis,
    legends=[
        "Airy",
        "Array",
        "Gaussian",
        f"Gaussian (width scaling={param_g_scaling.beam_width_scaling})",
    ],
    ax=ax,
)
ax.set_ylim(0, None)
plt.show()
