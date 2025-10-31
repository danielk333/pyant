#!/usr/bin/env python

"""
Test beam models
"""

import pytest
import numpy as np
import numpy.testing as nt

import pyant
import spacecoords.spherical as sph

grp_num = 10
ant_num = 5
antennas = np.zeros((3, ant_num, grp_num))
antx = np.linspace(-50, 50, grp_num)
anty = np.linspace(-50, 50, ant_num)
for ind in range(grp_num):
    antennas[0, :, ind] = antx[ind]
    antennas[1, :, ind] = anty

NUM = 11
zenith = np.array([0, 0, 1], dtype=np.float64)
pointing = sph.sph_to_cart(np.array([0, 80.0, 1]), degrees=True)
k_vector = np.ones((3, NUM), dtype=np.float64)
k_vector[0, :] = 0
k_vector[1, :] = np.linspace(-0.1, 0.1, NUM)


# array_beam = pyant.models.Array(
#     pointing=np.array([0, 0, 1], dtype=np.float64),
#     frequency=50e6,
#     antennas=antennas,
# )
# cas_beam = pyant.models.Cassegrain(
#     pointing=pointing,
#     frequency=930e6,
#     inner_radius=3.0,
#     outer_radius=23.0,
# )

MODEL_FACTORIES = []


def add_model_param_factory(func):
    MODEL_FACTORIES.append(func)
    return func


@add_model_param_factory
def make_airy(scalar_param: bool = True):
    beam = pyant.models.Airy()
    if scalar_param:
        param = pyant.models.AiryParams(
            pointing=pointing,
            frequency=930e6,
            radius=23.0,
        )
    else:
        param = pyant.models.AiryParams(
            pointing=np.broadcast_to(
                pointing.reshape(3, 1),
                (3, NUM),
            ).copy(),
            frequency=np.linspace(200e6, 930e6, num=NUM),
            radius=np.full((NUM,), 23.0, dtype=np.float64),
        )
    return beam, param


@add_model_param_factory
def make_cassegrain(scalar_param: bool = True):
    beam = pyant.models.Cassegrain()
    if scalar_param:
        param = pyant.models.CassegrainParams(
            pointing=pointing,
            frequency=930e6,
            inner_radius=3.0,
            outer_radius=23.0,
        )
    else:
        param = pyant.models.CassegrainParams(
            pointing=np.broadcast_to(
                pointing.reshape(3, 1),
                (3, NUM),
            ).copy(),
            frequency=np.linspace(200e6, 930e6, num=NUM),
            inner_radius=np.full((NUM,), 3.0, dtype=np.float64),
            outer_radius=np.full((NUM,), 23.0, dtype=np.float64),
        )
    return beam, param


# models_vector = [
#     pyant.models.Gaussian(
#         pointing=np.broadcast_to(
#             pointing.reshape(3, 1),
#             (3, num),
#         ).copy(),
#         frequency=np.linspace(200e6, 930e6, num=num),
#         radius=np.full((num,), 100.0, dtype=np.float64),
#         normal_pointing=np.broadcast_to(
#             zenith.reshape(3, 1),
#             (3, num),
#         ).copy(),
#     ),
#     pyant.models.MeasuredAzimuthallySymmetric(
#         pointing=np.broadcast_to(
#             pointing.reshape(3, 1),
#             (3, num),
#         ).copy(),
#         off_axis_angle=np.linspace(0, 90, 100),
#         gains=np.linspace(0, 1, 100),
#         interpolation_method="linear",
#         degrees=True,
#     ),
#     pyant.models.MeasuredAzimuthallySymmetric(
#         pointing=np.broadcast_to(
#             pointing.reshape(3, 1),
#             (3, num),
#         ).copy(),
#         off_axis_angle=np.linspace(0, 90, 100),
#         gains=np.linspace(0, 1, 100),
#         interpolation_method="cubic_spline",
#         degrees=True,
#     ),
#     pyant.models.Array(
#         pointing=np.broadcast_to(
#             pointing.reshape(3, 1),
#             (3, num),
#         ).copy(),
#         frequency=np.full((num,), 50e6, dtype=np.float64),
#         antennas=antennas,
#     ),
#     pyant.models.FiniteCylindricalParabola(
#         pointing=np.broadcast_to(
#             pointing.reshape(3, 1),
#             (3, num),
#         ).copy(),
#         frequency=np.full((num,), 50e6, dtype=np.float64),
#         height=np.full((num,), 100.0, dtype=np.float64),
#         width=np.full((num,), 40.0, dtype=np.float64),
#         aperture_width=np.full((num,), 80.0, dtype=np.float64),
#     ),
#     pyant.models.PhasedFiniteCylindricalParabola(
#         pointing=np.broadcast_to(
#             pointing.reshape(3, 1),
#             (3, num),
#         ).copy(),
#         frequency=np.full((num,), 30e6, dtype=np.float64),
#         width=np.full((num,), 120, dtype=np.float64),
#         height=np.full((num,), 40, dtype=np.float64),
#         aperture_width=np.full((num,), 120, dtype=np.float64),
#         phase_steering=np.linspace(0, 30, num=num),
#         depth=np.full((num,), 18, dtype=np.float64),
#         degrees=True,
#     ),
# ]
#
# models_scalar = [
#     pyant.models.Gaussian(
#         pointing=pointing,
#         frequency=45e6,
#         radius=100.0,
#         normal_pointing=zenith,
#     ),
#     pyant.models.Isotropic(),
#     pyant.models.MeasuredAzimuthallySymmetric(
#         pointing=pointing,
#         off_axis_angle=np.linspace(0, 90, 100),
#         gains=np.linspace(0, 1, 100),
#         interpolation_method="linear",
#         degrees=True,
#     ),
#     pyant.models.MeasuredAzimuthallySymmetric(
#         pointing=pointing,
#         off_axis_angle=np.linspace(0, 90, 100),
#         gains=np.linspace(0, 1, 100),
#         interpolation_method="cubic_spline",
#         degrees=True,
#     ),
#     pyant.models.Array(
#         pointing=np.array([0, 0, 1.0]),
#         frequency=46.5e6,
#         antennas=antennas,
#     ),
#     pyant.models.FiniteCylindricalParabola(
#         pointing=np.array([0, 0, 1], dtype=np.float64),
#         frequency=224.0e6,
#         width=120.0,
#         height=40.0,
#         aperture_width=120.0,
#     ),
#     pyant.models.PhasedFiniteCylindricalParabola(
#         pointing=np.array([0, 0, 1], dtype=np.float64),
#         phase_steering=0.1,
#         frequency=30e6,
#         width=120.0,
#         height=40.0,
#         depth=18.0,
#         aperture_width=120.0,
#     ),
# ]
#
# _beam = pyant.models.InterpolatedArray(pointing=array_beam.parameters["pointing"].copy())
# _beam.generate_interpolation(
#     array_beam, resolution=(400, 400, None), min_elevation=60.0, interpolate_channels=[0]
# )
# models_scalar.append(_beam)
#
# _beam = pyant.models.InterpolatedArray(pointing=array_beam.parameters["pointing"].copy())
# _beam.generate_interpolation(
#     array_beam, resolution=(50, 50, 100), min_elevation=70.0, interpolate_channels=[0]
# )
# models_scalar.append(_beam)
#
# _beam_base = pyant.models.Interpolated()
# _beam_base.generate_interpolation(
#     cas_beam,
#     resolution=150,
# )
# models_scalar.append(_beam_base)
#
# _beam = pyant.models.InterpolatedArray(
#     pointing=np.broadcast_to(
#         pointing.reshape(3, 1),
#         (3, num),
#     ).copy(),
# )
# _beam.generate_interpolation(
#     array_beam, resolution=(400, 400, None), min_elevation=60.0, interpolate_channels=[0]
# )
# models_vector.append(_beam)
#


def func_name(factory):
    return f"[{factory.__name__}]"


@pytest.mark.parametrize("factory", MODEL_FACTORIES, ids=func_name)
def test_param_size(factory):
    _, param_sc = factory(scalar_param=True)
    _, param_ve = factory(scalar_param=False)
    assert param_sc.size() is None
    assert param_ve.size() == NUM


@pytest.mark.parametrize("factory", MODEL_FACTORIES, ids=func_name)
def test_copy_beam(factory):
    beam, _ = factory(scalar_param=False)
    beam2 = beam.copy()
    assert id(beam2) != id(beam)


@pytest.mark.parametrize("factory", MODEL_FACTORIES, ids=func_name)
def test_copy_param_scalar(factory):
    _, param = factory(scalar_param=False)
    param2 = param.copy()
    assert id(param2) != id(param)

    for key in param.keys:
        p = getattr(param, key)
        p2 = getattr(param2, key)
        if isinstance(p, np.ndarray):
            assert id(p2) != id(p)
            nt.assert_array_almost_equal(p2, p)
        else:
            nt.assert_almost_equal(p2, p)


@pytest.mark.parametrize("factory", MODEL_FACTORIES, ids=func_name)
def test_copy_param_vector(factory):
    _, param = factory(scalar_param=True)
    param2 = param.copy()
    assert id(param2) != id(param)

    for key in param.keys:
        p = getattr(param, key)
        p2 = getattr(param2, key)
        if isinstance(p, np.ndarray):
            assert id(p2) != id(p)
            nt.assert_array_almost_equal(p2, p)
        else:
            nt.assert_almost_equal(p2, p)


#
# @pytest.mark.parametrize("beam", models_vector, ids=func_name)
# def test_single_k_vector_params(beam):
#     g = beam.gain(k_vector[:, 0])
#     assert g.shape == (num,)
#
#
# @pytest.mark.parametrize("beam", models_vector, ids=func_name)
# def test_many_k_vector_params(beam):
#     g = beam.gain(k_vector)
#     assert g.shape == (num,)
#
#
# @pytest.mark.parametrize("beam", models_scalar, ids=func_name)
# def test_single_k_scalar_params(beam):
#     g = beam.gain(k_vector[:, 0])
#     assert len(g.shape) == 0
#
#
# @pytest.mark.parametrize("beam", models_scalar, ids=func_name)
# def test_many_k_scalar_params(beam):
#     g = beam.gain(k_vector)
#     assert g.shape == (num,)
#
#
# # @pytest.mark.parametrize("beam", models_vector + models_scalar, ids=func_name)
# def test_save_load(beam):
#     raise NotImplementedError("todo")
#
#
# if __name__ == "__main__":
#     index = 2
#     test_many_k_vector_params(models_vector[index])
#     test_single_k_vector_params(models_vector[index])
#     test_many_k_scalar_params(models_scalar[index])
#     test_single_k_scalar_params(models_scalar[index])
