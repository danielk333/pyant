#!/usr/bin/env python

"""Useful coordinate related functions."""
from typing import Callable, Iterable
import numpy as np
import spacecoords.spherical as sph
import matplotlib.pyplot as plt
import matplotlib as mpl
import mpl_toolkits.mplot3d as mpl3d

from matplotlib import animation
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from .beam import Beam
from .types import Parameters, NDArray_3xN, NDArray_3xNxM, NDArray_3
from . import utils


def add_circle(ax, c, r, fmt="k--", *args, **kw):
    th = np.linspace(0, 2 * np.pi, 180)
    ax.plot(c[0] + np.cos(th), c[1] + np.sin(th), fmt, *args, **kw)


def antenna_configuration(
    antennas: list[NDArray_3xN] | NDArray_3xNxM,
    ax: plt.Axes | mpl3d.Axes3D | None = None,
    color: str | None = None,
    z_axis: bool = True,
) -> tuple[plt.Figure | None, plt.Axes]:
    """Plot the 3d antenna positions"""
    if ax is None:
        fig = plt.figure(figsize=(15, 7))
        if z_axis:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)
    else:
        fig = None

    z_axis = ax.name == "3d"

    if color is None:
        style_ = "."
    else:
        style_ = "." + color

    if isinstance(antennas, list):
        stacked_antennas = np.concatenate(antennas, axis=1)
    else:
        stacked_antennas = antennas.reshape(3, -1)

    if z_axis:
        ax.plot(
            stacked_antennas[0, :],
            stacked_antennas[1, :],
            stacked_antennas[2, :],
            style_,
        )
    else:
        ax.plot(
            stacked_antennas[0, :],
            stacked_antennas[1, :],
            style_,
        )
    ax.set_title("Antennas", fontsize=22)
    ax.set_xlabel("X-position [m]", fontsize=20)
    ax.set_ylabel("Y-position [m]", fontsize=20)
    if z_axis:
        ax.set_zlabel("Z-position [m]", fontsize=20)  # type: ignore

    return fig, ax


def gains(
    beams: list[Beam],
    params: list[Parameters],
    resolution: int = 1000,
    min_elevation: float = 0.0,
    max_elevation: float = 90.0,
    alpha: float = 1,
    legends: list[str] | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure | None, plt.Axes, list[plt.Line2D]]:
    """Plot the gain of a list of beam patterns as a function of elevation at
    `0^\\circ` degrees azimuth.

    Parameters
    ----------
    resolution
        Number of points to divide the set elevation range into.
    min_elevation
        Minimum elevation in degrees
    max_elevation
        Maximum elevation in degrees
    alpha
        The alpha with which to draw the curves
    legends
        Labels to put on each curve

    Returns
    -------
        Returns the matplotlib figure, axis and list of drawn lines
    """

    if ax is None:
        fig = plt.figure(figsize=(15, 7))
        ax = fig.add_subplot(111)
    else:
        fig = None

    theta = np.linspace(min_elevation, max_elevation, num=resolution)
    k_ang = np.zeros((3, resolution), dtype=np.float64)
    k_ang[1, :] = theta
    k_ang[2, :] = 1.0
    k = sph.sph_to_cart(k_ang, degrees=True)

    S = np.zeros((resolution, len(beams)))
    for bind, (beam, param) in enumerate(zip(beams, params)):
        S[:, bind] = beam.gain(k, param)
    lns = []
    for bind in range(len(beams)):
        lg = legends[bind] if legends is not None else None
        (ln,) = ax.plot(90 - theta, np.log10(S[:, bind]) * 10.0, alpha=alpha, label=lg)
        lns.append(ln)
    if legends is not None:
        ax.legend()

    ax.set_xlabel("Zenith angle [deg]")
    ax.tick_params(axis="both")
    ax.set_ylabel("Gain [dB]")
    ax.set_title("Gain patterns")

    return fig, ax, lns


def gain_surface(
    beam: Beam,
    param: Parameters,
    resolution: int = 201,
    min_elevation: float = 0.0,
    render_resolution: int | None = None,
    clip_low_dB: bool = True,
    ax: plt.Axes | None = None,
    label: str | None = None,
    centered: bool = True,
    cmap: plt.Colormap | None = None,
) -> tuple[plt.Figure | None, plt.Axes, mpl3d.art3d.Patch3DCollection]:
    """Creates a 3d plot of the beam-patters as a function of azimuth and
    elevation in terms of wave vector ground projection coordinates.

    Parameters
    ----------
    resolution
        Number of points to devide the wave vector x and y
        component range into, total number of caluclation points is the square of this number.
    min_elevation
        Minimum elevation in degrees, elevation range
        is from this number to `90^\\circ`. This number defines the half
        the length of the square that the gain is calculated over, i.e. `\\cos(el_{min})`.
    label
        Adds this to plot title
    centered
        Choose if plot is centered on pointing direction (:code:`True`) or zenith (:code:`False`)
    clip_low_dB
        If `True` set all gains below 0 dB to 0 dB

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

    if not isinstance(beam, Beam):
        raise TypeError(f"Can only plot Beam, not '{type(beam)}'")
    if param.size is None:
        raise ValueError(
            "Can only plot beam with scalar parameters -"
            f"dont know which of the {param.size} options to pick"
        )
    if "pointing" not in param.keys:
        pointing = np.array([0, 0, 1], dtype=np.float64)
    else:
        pointing = param.pointing  # type: ignore

    cmin = np.cos(np.radians(min_elevation))
    S, K, k, inds, kx, ky = utils.compute_k_grid(pointing, resolution, centered, cmin)

    S[inds] = beam.gain(k[:, inds], param)
    S = S.reshape(resolution, resolution)

    old = np.seterr(invalid="ignore")
    SdB = np.log10(S) * 10.0
    np.seterr(**old)

    if cmap is None:
        cmap = plt.get_cmap("plasma")

    rend_count = resolution if render_resolution is None else render_resolution

    if clip_low_dB:
        SdB[SdB < 0] = 0

    surf = ax.plot_surface(  # type: ignore
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
    ax.set_zlabel("Gain [dB]")  # type: ignore
    return fig, ax, surf


def polarization_heatmap(
    beam: Beam,
    param: Parameters,
    k: NDArray_3,
    resolution: int = 201,
    levels: int = 20,
    ax: plt.Axes | None = None,
    label: str | None = None,
    cmap: plt.Colormap | None = None,
) -> tuple[plt.Figure | None, plt.Axes, mpl.collections.QuadMesh]:
    """Creates a heatmap of the gain in the given direction as a function of polarization"""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if not isinstance(beam, Beam):
        raise TypeError(f"Can only plot Beam, not '{type(beam)}'")

    if param.size is None:
        raise ValueError(
            "Can only plot beam with scalar parameters -"
            f"dont know which of the {param.size} options to pick"
        )
    # We will draw a k-space circle centered on `pointing` with a radius of cos(min_elevation)
    jones_vecs, thxmat, thymat = utils.compute_j_grid(resolution)

    g = np.zeros((jones_vecs.shape[1],), dtype=np.float64)
    param_c = param.copy()
    for ind in range(jones_vecs.shape[1]):
        param_c.polarization = jones_vecs[:, ind]
        g[ind] = beam.gain(k, param_c)
    g = g.reshape(resolution, resolution)

    old = np.seterr(invalid="ignore")
    gdB = np.log10(g) * 10.0
    np.seterr(**old)

    if cmap is None:
        cmap = plt.get_cmap("plasma")

    if levels is None:
        conf = ax.pcolormesh(np.degrees(thxmat), np.degrees(thymat), gdB, cmap=cmap, vmin=0)
    else:
        # Recipe at
        # https://matplotlib.org/3.1.3/gallery/images_contours_and_fields/pcolormesh_levels.html
        bins = MaxNLocator(nbins=levels).tick_values(np.nanmin(gdB), np.nanmax(gdB))
        norm = BoundaryNorm(bins, ncolors=cmap.N, clip=True)
        conf = ax.pcolormesh(np.degrees(thxmat), np.degrees(thymat), gdB, cmap=cmap, norm=norm)

    ax.axis("scaled")
    ax.set_clip_box(ax.bbox)

    ax.set_xlabel("Jones theta_x [deg]")
    ax.set_ylabel("Jones theta_y [deg]")

    cbar = plt.colorbar(conf, ax=ax)
    cbar.ax.set_ylabel("Gain [dB]")
    tit = f"Gain for k=({k[0]:.2f},{k[1]:.2f},{k[2]:.2f}) as a function of polarization"
    if label:
        tit += " " + label
    ax.set_title(tit)

    return fig, ax, conf


def gain_heatmap(
    beam: Beam,
    param: Parameters,
    resolution: int = 201,
    min_elevation: float = 0.0,
    levels: int = 20,
    cbar_min: float | None = None,
    cbar_max: float | None = None,
    ax: plt.Axes | None = None,
    label: str | None = None,
    centered: bool = True,
    cmap: plt.Colormap | None = None,
) -> tuple[plt.Figure | None, plt.Axes, mpl.collections.QuadMesh]:
    """Creates a heatmap of the beam-patterns as a function of azimuth and
    elevation in terms of wave vector ground projection coordinates.

    # todo update docstring

    Parameters
    ----------
    beam
        Beam to plot
    resolution
        Number of points to devide the wave vector x and y
        component range into, total number of caluclation points is the square of this number.
    min_elevation
        Minimum elevation in degrees, elevation range
        is from this number to :math:`90^\\circ`. This number defines the half
        the length of the square that the gain is calculated over, i.e. :math:`\\cos(el_{min})`.
    label
        Adds this to plot title
    centered
        Choose if plot is centered on pointing direction (:code:`True`) or zenith (:code:`False`)
    levels
        Number of levels in the contour plot.
    cbar_min
        The minimum color (in dB) shown in the colorbar
    cbar_max
        The maximum color (in dB) shown in the colorbar

    Returns
    -------
    tuple(Figure, Axis, pcolormesh)
        Returns the matplotlib figure, axis and drawn pcolormesh

    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if not isinstance(beam, Beam):
        raise TypeError(f"Can only plot Beam, not '{type(beam)}'")
    if param.size is None:
        raise ValueError(
            "Can only plot beam with scalar parameters -"
            f"dont know which of the {param.size} options to pick"
        )
    if "pointing" not in param.keys:
        pointing = np.array([0, 0, 1], dtype=np.float64)
    else:
        pointing = param.pointing  # type: ignore

    # We will draw a k-space circle centered on `pointing` with a radius of cos(min_elevation)
    cmin = np.cos(np.radians(min_elevation))
    S, K, k, inds, kx, ky = utils.compute_k_grid(pointing, resolution, centered, cmin)

    S[inds] = beam.gain(k[:, inds], param)
    S = S.reshape(resolution, resolution)

    old = np.seterr(invalid="ignore")
    SdB = np.log10(S) * 10.0
    np.seterr(**old)

    if cmap is None:
        cmap = plt.get_cmap("plasma")

    if levels is None:
        conf = ax.pcolormesh(K[:, :, 0], K[:, :, 1], SdB, cmap=cmap, vmin=cbar_min, vmax=cbar_max)
    else:
        # Recipe at
        # https://matplotlib.org/3.1.3/gallery/images_contours_and_fields/pcolormesh_levels.html
        cmin = np.nanmin(SdB) if cbar_min is None else cbar_min
        cmax = np.nanmax(SdB) if cbar_max is None else cbar_max
        bins = MaxNLocator(nbins=levels).tick_values(cmin, cmax)
        norm = BoundaryNorm(bins, ncolors=cmap.N, clip=True)
        conf = ax.pcolormesh(K[:, :, 0], K[:, :, 1], SdB, cmap=cmap, norm=norm)

    ax.axis("scaled")
    ax.set_clip_box(ax.bbox)

    add_circle(ax, [0, 0], 1.0, "--", linewidth=1, color="#c0c0c0")
    add_circle(ax, pointing[:2], cmin, "-.", linewidth=1, color="#c0c0c0")

    ax.set_xlabel("kx [1]")
    ax.set_ylabel("ky [1]")

    cbar = plt.colorbar(conf, ax=ax)
    cbar.ax.set_ylabel("Gain [dB]")
    tit = "Gain pattern"
    if label:
        tit += " " + label
    ax.set_title(tit)

    return fig, ax, conf


def hemisphere_plot(
    func: Callable,
    plotfunc: Callable | str,
    preproc: str | None = "dba",
    f_args: list = [],
    f_kw: dict = {},
    p_args: list = [],
    p_kw: dict = {},
    resolution: int = 201,
    ax: plt.Axes | None = None,
    min_elevation: float = 0.0,
    centered: NDArray_3 | None = None,
) -> tuple[plt.Figure | None, plt.Axes, mpl.collections.QuadMesh]:
    """
    Create a hemispherical plot of some function of pointing direction

    Parameters
    ----------
    func
        Some function that maps from a pointing vector in the upper hemisphere to a scalar
    plotfunc
        a function with call signature like `contourf` or
        `pcolormesh`, i.e.  plotfunc(xval, yval, zval, *args, **kw)
    f_args
        extra arguments to `func`
    f_kw
        extra keyword arguments to `func`
    p_args
        extra arguments to `plotfunc`
    p_kw
        extra keyword arguments to `plotfunc`
    resolution
        Number of points to divide the wave vector x and y
        components into, total number of calculation points is the
        square of this number.
    plot_axis
        Axis in which to make the plot.
        If not given, one will be created in a new figure window
    preproc
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
    S, K, k, inds, kx, ky = utils.compute_k_grid(pointing, resolution, centered, cmin)

    S[inds] = func(k[:, inds]).flatten()
    S = S.reshape(resolution, resolution)

    # TODO: Some cleverness with try/except, perhaps?
    plotfunc_ = getattr(ax, plotfunc) if isinstance(plotfunc, str) else plotfunc

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

    hh = plotfunc_(K[:, :, 0], K[:, :, 1], S, *p_args, **p_kw)
    ax.axis("scaled")

    return fig, ax, hh


def gain_heatmap_movie(
    beam: Beam,
    param: Parameters,
    iterable: Iterable,
    param_update: Callable[[Parameters, int], Parameters],
    resolution: int = 201,
    min_elevation: float = 0.0,
    levels: int = 20,
    label: str | None = None,
    centered: bool = True,
    cmap: plt.Colormap | None = None,
    plot_update: (
        Callable[
            [plt.Figure, plt.Axes, mpl.collections.QuadMesh],
            tuple[plt.Figure, plt.Axes, mpl.collections.QuadMesh],
        ]
        | None
    ) = None,
    fps: int = 20,
    blit: bool = True,
):
    """
    Animates a movie of a heatmap
    """

    fig, ax, mesh = gain_heatmap(
        beam,
        param,
        resolution=resolution,
        min_elevation=min_elevation,
        levels=levels,
        label=label,
        centered=centered,
        cmap=cmap,
    )
    if fig is None:
        raise TypeError("Need a figure handle")

    if "pointing" not in param.keys:
        pointing = np.array([0, 0, 1], dtype=np.float64)
    else:
        pointing = param.pointing  # type: ignore

    cmin = np.cos(np.radians(min_elevation))
    S, K, k, inds, kx, ky = utils.compute_k_grid(pointing, resolution, centered, cmin)

    def run(it, fig, ax, mesh, beam, resolution, S, k, inds):
        new_param = param_update(param, it)
        S[inds] = beam.gain(k[:, inds], new_param).flatten()
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
        fargs=(fig, ax, mesh, beam, resolution, S, k, inds),
    )

    return fig, ax, mesh, ani
