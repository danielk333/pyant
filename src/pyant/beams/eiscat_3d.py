#!/usr/bin/env python

"""A collection of functions and information for the EISCAT 3D Radar system.

Notes
-----
Configuration are taken from [1]_.


.. [1] (Technical report) Vierinen, J., Kastinen, D., Kero, J.,
    Grydeland, T., McKay, D., Roynestad, E., Hesselbach, S., Kebschull, C., &
    Krag, H. (2019). EISCAT 3D Performance Analysis

"""

# Python standard import
import pathlib

import numpy as np
import scipy.constants

from ..models import Array, InterpolatedArray
from .beams import radar_beam_generator
from ..registry import Radars, Models
from .data import DATA

_data_file = DATA["e3d_subgroup_positions.txt"] if "e3d_subgroup_positions.txt" in DATA else None

e3d_frequency = 233e6
e3d_antenna_gain = 10.0**0.3  # 3 dB peak antenna gain?


def e3d_subarray(freqeuncy):
    """Generate cartesian positions `x,y,z` in meters of antenna elements in
    one standard EISCAT 3D subarray.
    """
    l0 = scipy.constants.c / freqeuncy

    dx = 1.0 / np.sqrt(3)
    dy = 0.5

    xall = []
    yall = []

    x0_p1 = np.arange(-2.5, -5.5, -0.5).tolist()
    x0_p2 = np.arange(-4.5, -2.0, 0.5).tolist()
    x0 = np.array([x0_p1 + x0_p2])[0] * dx
    y0 = np.arange(-5, 6, 1) * dy

    for iy in range(11):
        nx = 11 - np.abs(iy - 5)
        x_now = x0[iy] + np.array(range(nx)) * dx
        y_now = y0[iy] + np.array([0.0] * (nx))
        xall += x_now.tolist()
        yall += y_now.tolist()

    x = l0 * np.array(xall)
    y = l0 * np.array(yall)
    z = x * 0.0

    return x, y, z


def e3d_array(freqeuncy, fname=None, configuration="full"):
    """Generate the antenna positions for a EISCAT 3D Site based on submodule
    positions of a file.
    """

    def _read_e3d_submodule_pos(string_data):
        dat = []
        file = string_data.split("\n")
        for line in file:
            if len(line) == 0:
                continue
            dat.append(list(map(lambda x: float(x), line.split())))
        dat = np.array(dat)
        return dat

    fname = _data_file if fname is None else fname

    with open(fname, "r") as stream:
        _ant_data = stream.read()
    dat = _read_e3d_submodule_pos(_ant_data)

    sx, sy, sz = e3d_subarray(freqeuncy)

    if configuration == "full":
        pass
    elif configuration == "half-dense":
        dat = dat[(np.sum(dat**2.0, axis=1) < 27.0**2.0), :]
    elif configuration == "half-sparse":
        dat = dat[
            np.logical_or(
                np.logical_or(
                    np.logical_and(
                        np.sum(dat**2, axis=1) < 10**2, np.sum(dat**2, axis=1) > 7**2
                    ),
                    np.logical_and(
                        np.sum(dat**2, axis=1) < 22**2, np.sum(dat**2, axis=1) > 17**2
                    ),
                ),
                np.logical_and(
                    np.sum(dat**2, axis=1) < 36**2,
                    np.sum(dat**2, axis=1) > 30**2,
                ),
            ),
            :,
        ]
    elif configuration == "module":
        dat = np.zeros((1, 2))

    antennas = np.zeros((3, len(sx), dat.shape[0]), dtype=dat.dtype)
    for i in range(dat.shape[0]):
        for j in range(len(sx)):
            antennas[0, j, i] = sx[j] + dat[i, 0]
            antennas[1, j, i] = sy[j] + dat[i, 1]
            antennas[2, j, i] = sz[j]
    return antennas


@radar_beam_generator(Radars.EISCAT_3D_module, Models.Array)
def generate_eiscat_3d_module():
    """EISCAT 3D Gain pattern for single antenna sub-array."""
    return Array(
        azimuth=0.0,
        elevation=90.0,
        frequency=e3d_frequency,
        antennas=e3d_array(
            e3d_frequency,
            configuration="module",
        ),
        scaling=e3d_antenna_gain,
        degrees=True,
    )


@radar_beam_generator(Radars.EISCAT_3D_stage1, Models.Array)
def generate_eiscat_3d_stage1(configuration="dense"):
    """EISCAT 3D Gain pattern for a dense core of active sub-arrays,
    i.e stage 1 of development.

    Parameters
    ----------
    configuration : {'dense', 'sparse'}, optional
        Chooses how the stage1 antennas are distributed in the full array.

    """
    return Array(
        azimuth=0.0,
        elevation=90.0,
        frequency=e3d_frequency,
        antennas=e3d_array(
            e3d_frequency,
            configuration="half-" + configuration,
        ),
        scaling=e3d_antenna_gain,
        degrees=True,
    )


@radar_beam_generator(Radars.EISCAT_3D_stage2, Models.Array)
def generate_eiscat_3d_stage2():
    """EISCAT 3D Gain pattern for a full site of active sub-arrays,
    i.e stage 2 of development.

    """
    return Array(
        azimuth=0.0,
        elevation=90.0,
        frequency=e3d_frequency,
        antennas=e3d_array(
            e3d_frequency,
            configuration="full",
        ),
        scaling=e3d_antenna_gain,
        degrees=True,
    )


@radar_beam_generator(Radars.EISCAT_3D_stage1, Models.InterpolatedArray)
def generate_eiscat_3d_stage1_interp(path, configuration="dense", resolution=(1000, 1000, None)):
    """EISCAT 3D Gain pattern for a dense core of active sub-arrays,
    i.e stage 1 of development.

    Parameters
    ----------
    configuration : {'dense', 'sparse'}, optional
        Chooses how the stage1 antennas are distributed in the full array.

    """
    beam = InterpolatedArray(
        azimuth=0.0,
        elevation=90.0,
        frequency=e3d_frequency,
        scaling=e3d_antenna_gain,
        degrees=True,
    )
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    if path.is_file():
        beam.load(path)
    else:
        array = generate_eiscat_3d_stage1(configuration=configuration)
        beam.generate_interpolation(array, resolution=resolution)
        beam.save(path)
    return beam
