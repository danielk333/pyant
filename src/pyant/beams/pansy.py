#!/usr/bin/env python

"""A collection of functions and information for the PANSY Radar system.

"""
import pathlib
import numpy as np
import scipy.interpolate

from ..models import Array, InterpolatedArray
from ..coordinates import cart_to_sph
from .beams import radar_beam_generator
from ..registry import Radars, Models
from .data import DATA
from .radar_parameters import RADAR_PARAMETERS


_data_file = DATA["antpos.csv-20160121"] if "antpos.csv-20160121" in DATA else None
if _data_file is not None:
    # SerialNo. , GrpName, ModuleID, Ready , X(m), Y(m), Z(m)
    _pansy_antennas = np.genfromtxt(
        _data_file,
        skip_header=1,
        delimiter=",",
        dtype="i4, U3, U3, i4, f8, f8, f8",
    )
else:
    _pansy_antennas = None


_yagi_data_file = DATA["mu_yagi_gain.npz"] if "mu_yagi_gain.npz" in DATA else None
if _yagi_data_file is not None:
    _mu_yagi = np.load(_yagi_data_file)
else:
    _mu_yagi = None


@radar_beam_generator(Radars.PANSY, Models.Array)
def generate_pansy_radar():
    az = _mu_yagi["az_deg"].reshape(-1, 721)
    az -= 180
    el = _mu_yagi["el_deg"].reshape(-1, 721)
    gain_dB = _mu_yagi["gain_dB"].reshape(-1, 721)
    gain_dB = gain_dB - np.max(_mu_yagi["gain_dB"])

    interp = scipy.interpolate.RegularGridInterpolator(
        (az[0, :], el[:, 0]),
        gain_dB.T,
        bounds_error=False,
    )

    def yagi(k, polarization):
        sph = cart_to_sph(k, degrees=True)
        G = 10 ** (interp(sph[:2, :].T) / 10.0)
        return np.stack([G, G], axis=0)

    groups = np.unique(_pansy_antennas["f1"])
    grp_remove = ["DIV", "Z1", "Z2"]
    groups = [grp for grp in groups if grp not in grp_remove]

    antennas = []
    for ind, grp in enumerate(groups):
        select = np.logical_and(
            _pansy_antennas["f1"] == grp,
            _pansy_antennas["f3"] == 1,
        )
        subgroup = np.empty((3, np.sum(select)), dtype=np.float64)
        subgroup[0, :] = _pansy_antennas["f4"][select]
        subgroup[1, :] = _pansy_antennas["f5"][select]
        subgroup[2, :] = _pansy_antennas["f6"][select]
        antennas.append(subgroup)

    beam = Array(
        azimuth=0.0,
        elevation=90.0,
        frequency=RADAR_PARAMETERS[Radars.PANSY]["frequency"],
        antennas=antennas,
        scaling=RADAR_PARAMETERS[Radars.PANSY]["power_per_element"],
        degrees=True,
        antenna_element=yagi,
    )
    beam.meta["groups"] = groups
    return beam


@radar_beam_generator(Radars.PANSY, Models.InterpolatedArray)
def generate_interpolated_mu_radar(path=None, **interpolation_kwargs):
    beam = InterpolatedArray(
        azimuth=0.0,
        elevation=90.0,
        degrees=True,
    )
    if path is None:
        array = generate_pansy_radar()
        beam.generate_interpolation(array, **interpolation_kwargs)
        return beam

    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    if path.is_file():
        beam.load(path)
    else:
        array = generate_pansy_radar()
        beam.generate_interpolation(array, **interpolation_kwargs)
        beam.save(path)
    return beam
