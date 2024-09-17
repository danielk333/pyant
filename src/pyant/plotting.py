#!/usr/bin/env python

"""Useful coordinate related functions.
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from .beam import Beam
from . import coordinates as coord


def _clint(p, c, lim=1):
    """clip interval [p-c, p+c] to [-lim, lim] (lim=1 by default)"""
    return np.clip([p - c, p + c], -lim, lim)


def add_circle(ax, c, r, fmt="k--", *args, **kw):
    th = np.linspace(0, 2 * np.pi, 180)
    ax.plot(c[0] + np.cos(th), c[1] + np.sin(th), fmt, *args, **kw)


def compute_k_grid(pointing, resolution, centered, cmin):
    if centered:
        kx = np.linspace(*_clint(pointing[0], cmin), num=resolution)
        ky = np.linspace(*_clint(pointing[1], cmin), num=resolution)
    else:
        kx = np.linspace(-cmin, cmin, num=resolution)
        ky = np.linspace(-cmin, cmin, num=resolution)

    K = np.zeros((resolution, resolution, 2))

    # TODO: Refactor evaluation of function on a hemispherical domain to a function"
    K[:, :, 0], K[:, :, 1] = np.meshgrid(kx, ky, sparse=False, indexing="ij")
    size = resolution**2
    k = np.empty((3, size), dtype=np.float64)
    k[0, :] = K[:, :, 0].reshape(1, size)
    k[1, :] = K[:, :, 1].reshape(1, size)

    # circles in k space, centered on vertical and pointing, respectively
    z2 = k[0, :] ** 2 + k[1, :] ** 2
    z2_c = (pointing[0] - k[0, :]) ** 2 + (pointing[1] - k[1, :]) ** 2

    if centered:
        inds = np.logical_and(z2_c < cmin**2, z2 <= 1.0)
    else:
        inds = z2 < cmin**2
    not_inds = np.logical_not(inds)

    k[2, inds] = np.sqrt(1.0 - z2[inds])
    k[2, not_inds] = 0
    S = np.ones((size,)) * np.nan

    return S, K, k, inds, kx, ky


def antenna_configuration(antennas, ax=None, color=None, z_axis=True):
    """Plot the 3d antenna positions"""
    if ax is None:
        fig = plt.figure(figsize=(15, 7))
        if z_axis:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)
    else:
        fig = None

    if color is None:
        style_ = "."
    else:
        style_ = "." + color

    if z_axis:
        ax.plot(
            antennas[:, 0, :].flatten(),
            antennas[:, 1, :].flatten(),
            antennas[:, 2, :].flatten(),
            style_,
        )
    else:
        ax.plot(
            antennas[:, 0, :].flatten(),
            antennas[:, 1, :].flatten(),
            style_,
        )
    ax.set_title("Antennas", fontsize=22)
    ax.set_xlabel("X-position [m]", fontsize=20)
    ax.set_ylabel("Y-position [m]", fontsize=20)
    if z_axis:
        ax.set_zlabel("Z-position [m]", fontsize=20)

    return fig, ax


def gains(
    beams,
    inds=None,
    polarizations=None,
    resolution=1000,
    min_elevation=0.0,
    alpha=1,
    legends=None,
    ax=None,
):
    """Plot the gain of a list of beam patterns as a function of elevation at
    :math:`0^\\circ` degrees azimuth.

    Parameters
    ----------
    beam : Beam, list(Beam)
        Beam or list of beams.
    inds : dict, tuple, list(dict, tuple)
        Indexing of the beam instance, see :class:`pyant.Beam` for more details
    polarization : numpy.ndarray, list(numpy.ndarray)
        The Jones vector, see :class:`pyant.Beam` for more details
    resolution : int
        Number of points to divide the set elevation range into.
    min_elevation : float
        Minimum elevation in degrees, elevation range is from this number to :math:`90^\\circ`.
    alpha : float
        The alpha with which to draw the curves
    legends : list(str)
        Labels to put on each curve

    Returns
    -------
    tuple(Figure, Axis, list(Lines))
        Returns the matplotlib figure, axis and list of drawn lines
    """

    if not isinstance(beams, list):
        beams = [beams]
        inds = [inds]
        polarizations = [polarizations]

    if ax is None:
        fig = plt.figure(figsize=(15, 7))
        ax = fig.add_subplot(111)
    else:
        fig = None

    theta = np.linspace(min_elevation, 90.0, num=resolution)
    sph = np.zeros((3, resolution), dtype=np.float64)
    sph[1, :] = theta
    sph[2, :] = 1.0
    k = coord.sph_to_cart(sph, degrees=True)

    S = np.zeros((resolution, len(beams)))
    for b, beam in enumerate(beams):
        S[:, b] = beam.gain(k, polarization=polarizations[b], ind=inds[b]).flatten()
    lns = []
    for b in range(len(beams)):
        lg = legends[b] if legends is not None else None
        ln = ax.plot(90 - theta, np.log10(S[:, b]) * 10.0, alpha=alpha, label=lg)
        lns.append(ln)
    if legends is not None:
        ax.legend()

    ax.set_xlabel("Zenith angle [deg]", fontsize=24)
    ax.tick_params(axis="both", labelsize=17)
    ax.set_ylabel("Gain [dB]", fontsize=24)
    ax.set_title("Gain patterns", fontsize=28)

    return fig, ax, lns


def gain_surface(
    beam,
    polarization=None,
    resolution=201,
    min_elevation=0.0,
    render_resolution=None,
    clip_low_dB=True,
    ax=None,
    ind=None,
    label=None,
    centered=True,
    cmap=None,
):
    """Creates a 3d plot of the beam-patters as a function of azimuth and
    elevation in terms of wave vector ground projection coordinates.

    Parameters
    ----------
    beam : Beam
        Beam to plot
    inds : dict, tuple
        Indexing of the beam instance, see :class:`pyant.Beam` for more details
    polarization : numpy.ndarray
        The Jones vector, see :class:`pyant.Beam` for more details
    resolution : int
        Number of points to devide the wave vector x and y
        component range into, total number of caluclation points is the square of this number.
    min_elevation : float
        Minimum elevation in degrees, elevation range
        is from this number to :math:`90^\\circ`. This number defines the half
        the length of the square that the gain is calculated over, i.e. :math:`\\cos(el_{min})`.
    label : str
        Adds this to plot title
    centered : bool
        Choose if plot is centered on pointing direction (:code:`True`) or zenith (:code:`False`)
    clip_low_dB : bool
        If :code:`True` set all gains below 0 dB to 0 dB

    Returns
    -------
    tuple(Figure, Axis, surface)
        Returns the matplotlib figure, axis and drawn surface

    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = None

    if isinstance(beam, Beam):
        params, shape = beam.get_parameters(ind, named=True)
        pointing = params["pointing"]
    else:
        raise TypeError(f'Can only plot Beam, not "{type(beam)}"')

    cmin = np.cos(np.radians(min_elevation))
    S, K, k, inds, kx, ky = compute_k_grid(pointing, resolution, centered, cmin)

    S[inds] = beam.gain(k[:, inds], polarization=polarization, ind=ind).flatten()
    S = S.reshape(resolution, resolution)

    old = np.seterr(invalid="ignore")
    SdB = np.log10(S) * 10.0
    np.seterr(**old)

    if cmap is None:
        cmap = plt.get_cmap("plasma")

    rend_count = resolution if render_resolution is None else render_resolution

    if clip_low_dB:
        SdB[SdB < 0] = 0

    surf = ax.plot_surface(
        K[:, :, 0],
        K[:, :, 1],
        SdB,
        cmap=cmap,
        linewidth=0,
        antialiased=False,
        vmin=0,
        vmax=np.nanmax(SdB),
        rcount=rend_count,
        ccount=rend_count,
    )

    tit = "Gain pattern"
    if label:
        tit += " " + label
    ax.set_title(tit)

    ax.set_xlabel("kx [1]")
    ax.set_ylabel("ky [1]")
    ax.set_zlabel("Gain [dB]")
    return fig, ax, surf


def gain_heatmap(
    beam,
    polarization=None,
    resolution=201,
    min_elevation=0.0,
    levels=20,
    ax=None,
    ind=None,
    label=None,
    centered=True,
    cmap=None,
):
    """Creates a heatmap of the beam-patterns as a function of azimuth and
    elevation in terms of wave vector ground projection coordinates.


    Parameters
    ----------
    beam : Beam
        Beam to plot
    inds : dict, tuple
        Indexing of the beam instance, see :class:`pyant.Beam` for more details
    polarization : numpy.ndarray
        The Jones vector, see :class:`pyant.Beam` for more details
    resolution : int
        Number of points to devide the wave vector x and y
        component range into, total number of caluclation points is the square of this number.
    min_elevation : float
        Minimum elevation in degrees, elevation range
        is from this number to :math:`90^\\circ`. This number defines the half
        the length of the square that the gain is calculated over, i.e. :math:`\\cos(el_{min})`.
    label : str
        Adds this to plot title
    centered : bool
        Choose if plot is centered on pointing direction (:code:`True`) or zenith (:code:`False`)
    levels : int
        Number of levels in the contour plot.

    Returns
    -------
    tuple(Figure, Axis, pcolormesh)
        Returns the matplotlib figure, axis and drawn pcolormesh

    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if isinstance(beam, Beam):
        params, shape = beam.get_parameters(ind, named=True)
        pointing = params["pointing"]
    else:
        raise TypeError(f'Can only plot Beam, not "{type(beam)}"')

    # We will draw a k-space circle centered on `pointing` with a radius of cos(min_elevation)
    cmin = np.cos(np.radians(min_elevation))
    S, K, k, inds, kx, ky = compute_k_grid(pointing, resolution, centered, cmin)

    S[inds] = beam.gain(k[:, inds], polarization=polarization, ind=ind).flatten()
    S = S.reshape(resolution, resolution)

    old = np.seterr(invalid="ignore")
    SdB = np.log10(S) * 10.0
    np.seterr(**old)

    if cmap is None:
        cmap = plt.get_cmap("plasma")

    if levels is None:
        conf = ax.pcolormesh(K[:, :, 0], K[:, :, 1], SdB, cmap=cmap, vmin=0)
    else:
        # Recipe at
        # https://matplotlib.org/3.1.3/gallery/images_contours_and_fields/pcolormesh_levels.html
        bins = MaxNLocator(nbins=levels).tick_values(0, np.nanmax(SdB))
        norm = BoundaryNorm(bins, ncolors=cmap.N, clip=True)
        conf = ax.pcolormesh(K[:, :, 0], K[:, :, 1], SdB, cmap=cmap, norm=norm)

    ax.axis("scaled")
    ax.set_clip_box(ax.bbox)

    add_circle(ax, [0, 0], 1.0, "--", linewidth=1, color="#c0c0c0")
    add_circle(ax, pointing[:2], cmin, "-.", linewidth=1, color="#c0c0c0")

    ax.set_xlabel("kx [1]")
    ax.set_ylabel("ky [1]")

    plt.xticks()
    plt.yticks()
    cbar = plt.colorbar(conf, ax=ax)
    cbar.ax.set_ylabel("Gain [dB]")
    tit = "Gain pattern"
    if label:
        tit += " " + label
    ax.set_title(tit)

    return fig, ax, conf


def hemisphere_plot(
    func,
    plotfunc,
    preproc="dba",
    f_args=[],
    f_kw={},
    p_args=[],
    p_kw={},
    resolution=201,
    ax=None,
    min_elevation=0,
    centered=None,
):
    """
    Create a hemispherical plot of some function of pointing direction

    Parameters
    ----------
    func : callable
        Some function that maps from a pointing vector in the upper hemisphere to a scalar
    plotfunc : callable
        a function with call signature like `contourf` or
        `pcolormesh`, i.e.  plotfunc(xval, yval, zval, *args, **kw)
    f_args : list
        extra arguments to `func`
    f_kw : dict
        extra keyword arguments to `func`
    p_args : list
        extra arguments to `plotfunc`
    p_kw : dict
        extra keyword arguments to `plotfunc`
    resolution : int
        Number of points to divide the wave vector x and y
        components into, total number of calculation points is the
        square of this number.
    plot_axis : matplotlib.Axis
        Axis in which to make the plot.
        If not given, one will be created in a new figure window
    preproc : string
        in ['none', 'abs', 'dba', 'dbp']

    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    # We will draw a k-space circle centered on `pointing` with a radius of cos(min_elevation)
    if centered is None:
        pointing = np.array([0.0, 0.0, 1.0])
    else:
        pointing = centered

    # We will draw a k-space circle centered on `pointing` with a radius of cos(min_elevation)
    cmin = np.cos(np.radians(min_elevation))
    S, K, k, inds, kx, ky = compute_k_grid(pointing, resolution, centered, cmin)

    S[inds] = func(k[:, inds]).flatten()
    S = S.reshape(resolution, resolution)

    if isinstance(plotfunc, str):
        # TODO: Some cleverness with try/except, perhaps?
        plotfunc = getattr(ax, plotfunc)

    if preproc in [None, "none"]:
        pass
    elif preproc in ["abs"]:
        S = np.abs(S)
    elif preproc in ["dba", "dbp"]:
        mul = {"dba": 10, "dbp": 20}[preproc]
        old = np.seterr(invalid="ignore")
        SdB = mul * np.log10(S)
        np.seterr(**old)
        S = SdB
    else:
        print(f"preprocessor {preproc} unknown")

    hh = plotfunc(K[:, :, 0], K[:, :, 1], S, *p_args, **p_kw)
    ax.axis("scaled")

    return fig, ax, hh


def gain_heatmap_movie(
    beam,
    iterable,
    beam_update,
    polarization=None,
    resolution=201,
    min_elevation=0.0,
    levels=20,
    ax=None,
    ind=None,
    label=None,
    centered=True,
    cmap=None,
    plot_update=None,
    fps=20,
    blit=True,
):
    """
    Animates a movie of a heatmap
    """

    fig, ax, mesh = gain_heatmap(
        beam,
        polarization=polarization,
        resolution=resolution,
        min_elevation=min_elevation,
        levels=levels,
        ind=ind,
        label=label,
        centered=centered,
        cmap=cmap,
    )

    params, _ = beam.get_parameters(ind, named=True)
    pointing = params["pointing"]
    cmin = np.cos(np.radians(min_elevation))
    S, K, k, inds, kx, ky = compute_k_grid(pointing, resolution, centered, cmin)

    def run(it, fig, ax, mesh, beam, ind, polarization, resolution, S, k, inds):
        beam = beam_update(beam, it)
        S[inds] = beam.gain(k[:, inds], polarization=polarization, ind=ind).flatten()
        S = S.reshape(resolution, resolution)

        old = np.seterr(invalid="ignore")
        SdB = np.log10(S) * 10.0
        np.seterr(**old)
        mesh.update({"array": SdB.ravel()})

        if plot_update is not None:
            fig, ax, mesh = plot_update(fig, ax, mesh)

        return [mesh]

    ani = animation.FuncAnimation(
        fig,
        run,
        iterable,
        blit=blit,
        interval=1.0e3 / float(fps),
        repeat=True,
        fargs=(fig, ax, mesh, beam, ind, polarization, resolution, S, k, inds),
    )

    return fig, ax, mesh, ani
