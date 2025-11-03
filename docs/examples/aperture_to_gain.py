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
from matplotlib import animation
import spacecoords.spherical as sph
import spacecoords.linalg as linalg
import pyant

# +
# # The basics
#
# To gain an understanding about the most commonly used formulas for the gain pattern of different
# radars we need to start by looking at a typical case and build up from there.
#
# So lets start with a sketch of a parabolic receiver dish antenna with a un-obstructing
# collecting antenna at the focus

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
# This sketch shows how a radar dish can be thought of as a planar aperture. The particular question
# we want to ask is: what is the signal strength measured if we collect all radiation that enters
# the aperture and sum it up as a function of incoming wave direction?
#
# This question can be asked because of the definition of a parabola where all distances between the
# focus, the reflecting surface, and the input aperture are the same by definition. Hence this
# collecting strategy is essentially the same as summing the wave at each point along the aperture.
#
# For example, what would the signal strength be of this incoming wave?


fig, ax = plt.subplots()

ax.plot(xs, ys, ls="-", c="k", label="reflector")
ax.plot(xs, ap, ls="--", c="b", label="aperture")
ax.plot([0], [focus_height], "ok", label="focus")

ray_xs = np.linspace(-width / 2 * 0.9, width / 2 * 0.9, num=line_num)
for ind in range(len(ray_xs)):
    rxs, rys = ray(ray_xs[ind], ray_angle=np.radians(5))
    ax.plot(rxs, rys, ls="-", c="r", alpha=0.5)

ax.legend()
ax.axis("equal")
plt.show()

# +
# Now already we see a small problem, the radiation is no longer collected perfectly at the focus.
# However, this is not an accurate picture of what is actually going on, in reality it would be more
# like this.


def sph_wave_source(x0, y0, wavelength, x, y, initial_phase):
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    return np.exp(-1j * (2 * np.pi / wavelength * r + initial_phase))


def e_field(x, y, k_hat, wavelength, res=1000):
    e0 = np.zeros(x.shape, dtype=np.complex128)
    inds = np.logical_and.reduce(
        [
            x > -width / 2,
            x < width / 2,
            y > parabolic_func(x),
            y < y_max,
        ]
    )
    source_xs = np.linspace(-width / 2, width / 2, num=res)
    for ind in range(res):
        y0 = parabolic_func(source_xs[ind])
        r = np.array([source_xs[ind], y0])
        phi0 = 2 * np.pi / wavelength * np.dot(k_hat, r)
        e0[inds] += sph_wave_source(source_xs[ind], y0, wavelength, x[inds], y[inds], phi0)

    return np.abs(e0).astype(np.float64)


def draw_e_wave_dish(ax, k_hat):
    ax.clear()
    ax.plot(xs, ys, ls="-", c="k", label="reflector")
    ax.plot(xs, ap, ls="--", c="b", label="aperture")
    ax.plot([0], [focus_height], "ok", label="focus")

    map_res = [200, 200]
    xmat, ymat = np.meshgrid(
        np.linspace(-width / 2, width / 2, num=map_res[0]),
        np.linspace(0, y_max, num=map_res[1]),
    )
    e_res = 200
    e_f = e_field(xmat.flatten(), ymat.flatten(), k_hat, wavelength, res=e_res)
    e_f = e_f.reshape(xmat.shape)

    ax.pcolormesh(xmat, ymat, e_f ** 2)

    source_xs = np.linspace(-width / 2, width / 2, num=e_res)
    ax.plot(source_xs, parabolic_func(source_xs), "x", ls="none", c="k", label="sources")
    for ind in range(len(ray_xs)):
        ax.plot(
            [ray_xs[ind], ray_xs[ind] - k_hat[0] * start_ray / 5],
            [y_max, y_max - k_hat[1] * start_ray / 5],
            ls="-",
            c="r",
        )

    ax.legend()
    ax.axis("equal")


k_vec = linalg.rot_mat_2d(-5, degrees=True) @ np.array([0, -1])

fig, ax = plt.subplots()
draw_e_wave_dish(ax, k_vec)
plt.show()


# +
# Lets see how this looks animated!

angs = np.linspace(0, -10, 100)


def update(ind):
    k_vec = linalg.rot_mat_2d(angs[ind], degrees=True) @ np.array([0, -1])
    draw_e_wave_dish(ax, k_vec)


fig, ax = plt.subplots()

ani = animation.FuncAnimation(
    fig,
    update,
    range(len(angs)),
    interval=1.0e3 / float(30),
    repeat=True,
)
plt.show()

# +
# Lets see how this looks like as a line plot of intensity as a function of incoming angle

zenith_ang = np.linspace(0, 5, num=100)
power = np.zeros_like(zenith_ang)
for ind in range(len(zenith_ang)):
    k_vec = linalg.rot_mat_2d(-zenith_ang[ind], degrees=True) @ np.array([0, -1])
    power[ind] = e_field(np.array([0]), np.array([focus_height]), k_vec, wavelength, res=1000)[0]
power = (power / np.max(power)) ** 2

beam = pyant.models.Airy()
param = pyant.models.AiryParams(
    pointing=np.array([0, 0, 1], dtype=np.float64),
    frequency=freq,
    radius=width / 2,
)
g = beam.sph_gain(
    azimuth=np.zeros_like(zenith_ang),
    elevation=90 - zenith_ang,
    parameters=param,
    degrees=True,
)

fig, ax = plt.subplots()
ax.plot(zenith_ang, 10 * np.log10(power), "-r")
ax.plot(zenith_ang, 10 * np.log10(g), "-g")
plt.show()

# Given a radar with some kind of aperture, defined by continuous sets, located on a flat plane,
# the resulting gain function can be described by a 2D Fourier transform of that aperture.
#
