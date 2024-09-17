#!/usr/bin/env python

"""A collection of functions and information for the MU Radar system.

"""
import pathlib
import numpy as np
import scipy.interpolate

from ..models import Array, InterpolatedArray
from .beams import radar_beam_generator
from ..registry import Radars, Models
from .data import DATA
from .radar_parameters import RADAR_PARAMETERS

_data_file = DATA["MU_antenna_pos.npy"] if "MU_antenna_pos.npy" in DATA else None
if _data_file is not None:
    _mu_antennas = np.load(_data_file)
else:
    _mu_antennas = None

_yagi_data_file = DATA["mu_yagi_gain.npz"] if "mu_yagi_gain.npz" in DATA else None
if _yagi_data_file is not None:
    _mu_yagi = np.load(_yagi_data_file)
else:
    _mu_yagi = None


def generate_yagi_pattern():
    _ = scipy.interpolate.RectBivariateSpline(
        _mu_yagi["az_deg"],
        _mu_yagi["el_deg"],
        _mu_yagi["gain_dB"],
    )
    raise NotImplementedError()


@radar_beam_generator(Radars.MU, Models.Array)
def generate_mu_radar():
    return Array(
        azimuth=0.0,
        elevation=90.0,
        frequency=RADAR_PARAMETERS[Radars.MU]["frequency"],
        antennas=_mu_antennas,
        scaling=RADAR_PARAMETERS[Radars.MU]["power_per_element"],
        degrees=True,
    )


@radar_beam_generator(Radars.MU, Models.InterpolatedArray)
def generate_interpolated_mu_radar(path, resolution=(1000, 1000, None)):
    beam = InterpolatedArray(
        azimuth=0.0,
        elevation=90.0,
        frequency=RADAR_PARAMETERS[Radars.MU]["frequency"],
        scaling=RADAR_PARAMETERS[Radars.MU]["power_per_element"],
        degrees=True,
    )
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    if path.is_file():
        beam.load(path)
    else:
        array = generate_mu_radar()
        beam.generate_interpolation(array, resolution=resolution)
        beam.save(path)
    return beam
