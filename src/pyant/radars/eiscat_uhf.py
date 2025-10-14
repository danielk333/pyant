#!/usr/bin/env python

"""
References for the data:

#TODO
"""
from typing import Literal
import numpy as np
from pyant.models.measured import MeasuredAzimuthallySymmetric, InterpMethods
from pyant.models.cassegrain import Cassegrain
from .station import RadarStation
from .data import DATA_FILES

BeamType = Literal["measured", "cassegrain"]


def _make_eiscat_uhf_beam(interpolation_method: InterpMethods) -> MeasuredAzimuthallySymmetric:
    assert "eiscat_uhf_bp.txt" in DATA_FILES, "data file missing!"
    data_file = DATA_FILES["eiscat_uhf_bp.txt"]

    with data_file.open("r") as stream:
        eiscat_beam_data = np.genfromtxt(stream)

    peak_gain = 10**4.81
    beam = MeasuredAzimuthallySymmetric(
        pointing=np.array([0, 0, 1], dtype=np.float64),
        off_axis_angle=eiscat_beam_data[:, 0],
        gains=10.0 ** (eiscat_beam_data[:, 1] / 10.0) * peak_gain,
        interpolation_method=interpolation_method,
        degrees=True,
    )
    return beam


def _make_eiscat_uhf_cassegrain_beam():
    beam = Cassegrain(
        pointing=np.array([0, 0, 1], dtype=np.float64),
        frequency=930e6,
        outer_radius=40.0,
        inner_radius=23.0,
        peak_gain=10**4.81,
    )
    return beam


def generate_eiscat_uhf_tromso(
    beam_type: BeamType = "measured", interpolation_method: InterpMethods = "linear"
) -> RadarStation:
    """Generate the EISCAT UHF tromso station"""
    if beam_type == "measured":
        beam = _make_eiscat_uhf_beam(interpolation_method)
    elif beam_type == "cassegrain":
        beam = _make_eiscat_uhf_cassegrain_beam()

    st = RadarStation(
        uid="eiscat_uhf_tromso",
        transmitter=True,
        receiver=True,
        beam=beam,
        frequency=930.0e6,
        power=1.6e6,
        noise_temperature=100,
        min_elevation=30,
        lat=69.58638888888889,
        lon=19.227222222222224,
        alt=86.0,
    )
    return st


def generate_eiscat_uhf_kiruna(
    beam_type: BeamType = "measured", interpolation_method: InterpMethods = "linear"
) -> RadarStation:
    """Generate the EISCAT UHF kiruna station"""
    if beam_type == "measured":
        beam = _make_eiscat_uhf_beam(interpolation_method)
    elif beam_type == "cassegrain":
        beam = _make_eiscat_uhf_cassegrain_beam()

    st = RadarStation(
        uid="eiscat_uhf_kiruna",
        transmitter=False,
        receiver=True,
        beam=beam,
        frequency=930.0e6,
        noise_temperature=100,
        min_elevation=30,
        lat=67.86055555555555,
        lon=20.435277777777777,
        alt=418.0,
    )
    return st


def generate_eiscat_uhf_sodankyla(
    beam_type: BeamType = "measured", interpolation_method: InterpMethods = "linear"
) -> RadarStation:
    """Generate the EISCAT UHF sodankyla station"""
    if beam_type == "measured":
        beam = _make_eiscat_uhf_beam(interpolation_method)
    elif beam_type == "cassegrain":
        beam = _make_eiscat_uhf_cassegrain_beam()

    st = RadarStation(
        uid="eiscat_uhf_sodankyla",
        transmitter=False,
        receiver=True,
        beam=beam,
        frequency=930.0e6,
        noise_temperature=100,
        min_elevation=30,
        lat=67.36361111111111,
        lon=26.626944444444444,
        alt=197.0,
    )
    return st
