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


@add_model_param_factory
def make_gaussian(scalar_param: bool = True):
    beam = pyant.models.Gaussian()
    if scalar_param:
        param = pyant.models.GaussianParams(
            pointing=pointing,
            normal_pointing=pointing,
            frequency=930e6,
            radius=20,
            beam_width_scaling=1,
        )
    else:
        param = pyant.models.GaussianParams(
            pointing=np.broadcast_to(
                pointing.reshape(3, 1),
                (3, NUM),
            ).copy(),
            normal_pointing=np.broadcast_to(
                pointing.reshape(3, 1),
                (3, NUM),
            ).copy(),
            frequency=np.linspace(200e6, 930e6, num=NUM),
            radius=np.full((NUM,), 20.0, dtype=np.float64),
            beam_width_scaling=np.ones((NUM,), dtype=np.float64),
        )
    return beam, param


@add_model_param_factory
def make_isotropic(scalar_param: bool = True):
    beam = pyant.models.Isotropic()
    if scalar_param:
        param = pyant.models.IsotropicParams()
    else:
        param = None
    return beam, param


def make_measured_az_sym(interpolation_method, scalar_param: bool = True):
    beam = pyant.models.MeasuredAzimuthallySymmetric(
        off_axis_angle=np.linspace(0, 90, 100),
        gains=np.linspace(0, 1, 100),
        interpolation_method=interpolation_method,
        degrees=True,
    )
    if scalar_param:
        param = pyant.models.MeasuredAzimuthallySymmetricParams(
            pointing=pointing,
        )
    else:
        param = pyant.models.MeasuredAzimuthallySymmetricParams(
            pointing=np.broadcast_to(
                pointing.reshape(3, 1),
                (3, NUM),
            ).copy(),
        )
    return beam, param


@add_model_param_factory
def make_measured_az_sym_lin(scalar_param: bool = True):
    return make_measured_az_sym(
        interpolation_method="linear",
        scalar_param=scalar_param,
    )


@add_model_param_factory
def make_measured_az_sym_spline(scalar_param: bool = True):
    return make_measured_az_sym(
        interpolation_method="cubic_spline",
        scalar_param=scalar_param,
    )


@add_model_param_factory
def make_gaussian_interpolation(scalar_param: bool = True):
    beam = pyant.models.Gaussian()
    param = pyant.models.GaussianParams(
        pointing=np.array([0, 0, 1], dtype=np.float64),
        normal_pointing=np.array([0, 0, 1], dtype=np.float64),
        frequency=46.5e6,
        radius=100.0,
        beam_width_scaling=1,
    )
    interp_beam = pyant.models.Interpolated()
    interp_beam.generate_interpolation(beam, param, resolution=100)
    if scalar_param:
        interp_param = pyant.models.InterpolatedParams()
    else:
        interp_param = None
    return interp_beam, interp_param


@add_model_param_factory
def make_array_interpolation(scalar_param: bool = True):
    beam = pyant.models.Array(
        antennas=antennas,
    )
    param = pyant.models.ArrayParams(
        pointing=np.array([0, 0, 1], dtype=np.float64),
        frequency=50e6,
        polarization=beam.polarization.copy(),
    )

    interp_beam = pyant.models.InterpolatedArray()
    interp_beam.generate_interpolation(beam, param, min_elevation=80, resolution=(100, 100))
    if scalar_param:
        interp_param = pyant.models.InterpolatedArrayParams(
            pointing=np.array([0, 0, 1], dtype=np.float64),
        )
    else:
        interp_param = pyant.models.InterpolatedArrayParams(
            pointing=np.broadcast_to(
                pointing.reshape(3, 1),
                (3, NUM),
            ).copy(),
        )

    return interp_beam, interp_param


@add_model_param_factory
def make_array(scalar_param: bool = True):
    beam = pyant.models.Array(
        antennas=antennas,
    )
    if scalar_param:
        param = pyant.models.ArrayParams(
            pointing=np.array([0, 0, 1], dtype=np.float64),
            frequency=50e6,
            polarization=beam.polarization.copy(),
        )
    else:
        param = pyant.models.ArrayParams(
            pointing=np.broadcast_to(
                pointing.reshape(3, 1),
                (3, NUM),
            ).copy(),
            frequency=np.linspace(200e6, 930e6, num=NUM),
            polarization=np.broadcast_to(
                beam.polarization.copy().reshape(2, 1),
                (2, NUM),
            ).copy(),
        )
    return beam, param


@add_model_param_factory
def make_cyl_par(scalar_param: bool = True):
    beam = pyant.models.FiniteCylindricalParabola()
    if scalar_param:
        param = pyant.models.FiniteCylindricalParabolaParams(
            pointing=np.array([0, 0, 1], dtype=np.float64),
            frequency=224.0e6,
            width=120.0,
            height=40.0,
            aperture_width=120.0,
        )
    else:
        param = pyant.models.FiniteCylindricalParabolaParams(
            pointing=np.broadcast_to(
                pointing.reshape(3, 1),
                (3, NUM),
            ).copy(),
            frequency=np.full((NUM,), 50e6, dtype=np.float64),
            height=np.full((NUM,), 100.0, dtype=np.float64),
            width=np.full((NUM,), 40.0, dtype=np.float64),
            aperture_width=np.full((NUM,), 80.0, dtype=np.float64),
        )
    return beam, param


@add_model_param_factory
def make_phased_cyl_par(scalar_param: bool = True):
    beam = pyant.models.PhasedFiniteCylindricalParabola()
    if scalar_param:
        param = pyant.models.PhasedFiniteCylindricalParabolaParams(
            pointing=np.array([0, 0, 1], dtype=np.float64),
            phase_steering=0.1,
            frequency=30e6,
            width=120.0,
            height=40.0,
            depth=18.0,
            aperture_width=120.0,
        )
    else:
        param = pyant.models.PhasedFiniteCylindricalParabolaParams(
            pointing=np.broadcast_to(
                pointing.reshape(3, 1),
                (3, NUM),
            ).copy(),
            frequency=np.full((NUM,), 30e6, dtype=np.float64),
            width=np.full((NUM,), 120, dtype=np.float64),
            height=np.full((NUM,), 40, dtype=np.float64),
            aperture_width=np.full((NUM,), 120, dtype=np.float64),
            phase_steering=np.linspace(0, 30, num=NUM),
            depth=np.full((NUM,), 18, dtype=np.float64),
        )
    return beam, param


def func_name(factory):
    return f"[{factory.__name__}]"


@pytest.mark.parametrize("factory", MODEL_FACTORIES, ids=func_name)
def test_param_size(factory):
    _, param_sc = factory(scalar_param=True)
    _, param_ve = factory(scalar_param=False)
    assert param_sc.size() is None
    if param_ve is not None:
        assert param_ve.size() == NUM


@pytest.mark.parametrize("factory", MODEL_FACTORIES, ids=func_name)
def test_copy_beam(factory):
    beam, _ = factory(scalar_param=False)
    beam2 = beam.copy()
    assert id(beam2) != id(beam)


@pytest.mark.parametrize("factory", MODEL_FACTORIES, ids=func_name)
def test_copy_param_scalar(factory):
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


@pytest.mark.parametrize("factory", MODEL_FACTORIES, ids=func_name)
def test_copy_param_vector(factory):
    _, param = factory(scalar_param=False)
    if param is None:
        return
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
def test_single_k_vector_params(factory):
    beam, param = factory(scalar_param=False)
    if param is None:
        return
    g = beam.gain(k_vector[:, 0], param)
    assert g.shape == (NUM,)


@pytest.mark.parametrize("factory", MODEL_FACTORIES, ids=func_name)
def test_many_k_vector_params(factory):
    beam, param = factory(scalar_param=False)
    if param is None:
        return
    g = beam.gain(k_vector, param)
    assert g.shape == (NUM,)


@pytest.mark.parametrize("factory", MODEL_FACTORIES, ids=func_name)
def test_single_k_scalar_params(factory):
    beam, param = factory(scalar_param=True)
    g = beam.gain(k_vector[:, 0], param)
    assert len(g.shape) == 0


@pytest.mark.parametrize("factory", MODEL_FACTORIES, ids=func_name)
def test_many_k_scalar_params(factory):
    beam, param = factory(scalar_param=True)
    g = beam.gain(k_vector, param)
    assert g.shape == (NUM,)


# @pytest.mark.parametrize("beam", models_vector + models_scalar, ids=func_name)
# def test_save_load(beam):
#     raise NotImplementedError("todo")
