#!/usr/bin/env python

"""
Test beam models
"""

import pytest
import numpy as np
import numpy.testing as nt

import pyant

grp_num = 10
ant_num = 5
antennas = np.zeros((3, ant_num, grp_num))
antx = np.linspace(-50, 50, grp_num)
anty = np.linspace(-50, 50, ant_num)
for ind in range(grp_num):
    antennas[0, :, ind] = antx[ind]
    antennas[1, :, ind] = anty

num = 11
zenith = np.array([0, 0, 1], dtype=np.float64)
pointing = pyant.coordinates.sph_to_cart(np.array([0, 80.0, 1]), degrees=True)
k_vector = np.ones((3, num), dtype=np.float64)
k_vector[0, :] = 0
k_vector[1, :] = np.linspace(-0.1, 0.1, num)


array_beam = pyant.models.Array(
    pointing=np.array([0, 0, 1], dtype=np.float64),
    frequency=50e6,
    antennas=antennas,
)
cas_beam = pyant.models.Cassegrain(
    pointing=pointing,
    frequency=930e6,
    inner_radius=3.0,
    outer_radius=23.0,
)


models_vector = [
    pyant.models.Airy(
        pointing=np.broadcast_to(
            pointing.reshape(3, 1),
            (3, num),
        ).copy(),
        frequency=np.linspace(200e6, 930e6, num=num),
        radius=np.full((num,), 23.0, dtype=np.float64),
    ),
    pyant.models.Cassegrain(
        pointing=np.broadcast_to(
            pointing.reshape(3, 1),
            (3, num),
        ).copy(),
        frequency=np.linspace(200e6, 930e6, num=num),
        inner_radius=np.full((num,), 3.0, dtype=np.float64),
        outer_radius=np.full((num,), 23.0, dtype=np.float64),
    ),
    pyant.models.Gaussian(
        pointing=np.broadcast_to(
            pointing.reshape(3, 1),
            (3, num),
        ).copy(),
        frequency=np.linspace(200e6, 930e6, num=num),
        radius=np.full((num,), 100.0, dtype=np.float64),
        normal_pointing=np.broadcast_to(
            zenith.reshape(3, 1),
            (3, num),
        ).copy(),
    ),
    pyant.models.MeasuredAzimuthallySymmetric(
        pointing=np.broadcast_to(
            pointing.reshape(3, 1),
            (3, num),
        ).copy(),
        off_axis_angle=np.linspace(0, 90, 100),
        gains=np.linspace(0, 1, 100),
        interpolation_method="linear",
        degrees=True,
    ),
    pyant.models.MeasuredAzimuthallySymmetric(
        pointing=np.broadcast_to(
            pointing.reshape(3, 1),
            (3, num),
        ).copy(),
        off_axis_angle=np.linspace(0, 90, 100),
        gains=np.linspace(0, 1, 100),
        interpolation_method="cubic_spline",
        degrees=True,
    ),
    pyant.models.Array(
        pointing=np.broadcast_to(
            pointing.reshape(3, 1),
            (3, num),
        ).copy(),
        frequency=np.full((num,), 50e6, dtype=np.float64),
        antennas=antennas,
    ),
    pyant.models.FiniteCylindricalParabola(
        pointing=np.broadcast_to(
            pointing.reshape(3, 1),
            (3, num),
        ).copy(),
        frequency=np.full((num,), 50e6, dtype=np.float64),
        height=np.full((num,), 100.0, dtype=np.float64),
        width=np.full((num,), 40.0, dtype=np.float64),
        aperture_width=np.full((num,), 80.0, dtype=np.float64),
    ),
    pyant.models.PhasedFiniteCylindricalParabola(
        pointing=np.broadcast_to(
            pointing.reshape(3, 1),
            (3, num),
        ).copy(),
        frequency=np.full((num,), 30e6, dtype=np.float64),
        width=np.full((num,), 120, dtype=np.float64),
        height=np.full((num,), 40, dtype=np.float64),
        aperture_width=np.full((num,), 120, dtype=np.float64),
        phase_steering=np.linspace(0, 30, num=num),
        depth=np.full((num,), 18, dtype=np.float64),
        degrees=True,
    ),
]

models_scalar = [
    pyant.models.Airy(
        pointing=pointing,
        frequency=930e6,
        radius=23.0,
    ),
    pyant.models.Cassegrain(
        pointing=pointing,
        frequency=930e6,
        inner_radius=3.0,
        outer_radius=23.0,
    ),
    pyant.models.Gaussian(
        pointing=pointing,
        frequency=45e6,
        radius=100.0,
        normal_pointing=zenith,
    ),
    pyant.models.Isotropic(),
    pyant.models.MeasuredAzimuthallySymmetric(
        pointing=pointing,
        off_axis_angle=np.linspace(0, 90, 100),
        gains=np.linspace(0, 1, 100),
        interpolation_method="linear",
        degrees=True,
    ),
    pyant.models.MeasuredAzimuthallySymmetric(
        pointing=pointing,
        off_axis_angle=np.linspace(0, 90, 100),
        gains=np.linspace(0, 1, 100),
        interpolation_method="cubic_spline",
        degrees=True,
    ),
    pyant.models.Array(
        pointing=np.array([0, 0, 1.0]),
        frequency=46.5e6,
        antennas=antennas,
    ),
    pyant.models.FiniteCylindricalParabola(
        pointing=np.array([0, 0, 1], dtype=np.float64),
        frequency=224.0e6,
        width=120.0,
        height=40.0,
        aperture_width=120.0,
    ),
    pyant.models.PhasedFiniteCylindricalParabola(
        pointing=np.array([0, 0, 1], dtype=np.float64),
        phase_steering=0.1,
        frequency=30e6,
        width=120.0,
        height=40.0,
        depth=18.0,
        aperture_width=120.0,
    ),
]

_beam = pyant.models.InterpolatedArray(pointing=array_beam.parameters["pointing"].copy())
_beam.generate_interpolation(
    array_beam, resolution=(400, 400, None), min_elevation=60.0, interpolate_channels=[0]
)
models_scalar.append(_beam)

_beam = pyant.models.InterpolatedArray(pointing=array_beam.parameters["pointing"].copy())
_beam.generate_interpolation(
    array_beam, resolution=(50, 50, 100), min_elevation=70.0, interpolate_channels=[0]
)
models_scalar.append(_beam)

_beam_base = pyant.models.Interpolated()
_beam_base.generate_interpolation(
    cas_beam,
    resolution=150,
)
models_scalar.append(_beam_base)

_beam = pyant.models.InterpolatedArray(
    pointing=np.broadcast_to(
        pointing.reshape(3, 1),
        (3, num),
    ).copy(),
)
_beam.generate_interpolation(
    array_beam, resolution=(400, 400, None), min_elevation=60.0, interpolate_channels=[0]
)
models_vector.append(_beam)


def beam_name(beam):
    return f"[{beam.__class__.__name__}]"


@pytest.mark.parametrize("beam", models_vector + models_scalar, ids=beam_name)
def test_copy(beam):
    b2 = beam.copy()
    for key, p in beam.parameters.items():
        assert key in b2.parameters
        if isinstance(p, np.ndarray):
            nt.assert_array_almost_equal(p, b2.parameters[key])
        else:
            nt.assert_almost_equal(p, b2.parameters[key])


@pytest.mark.parametrize("beam", models_vector, ids=beam_name)
def test_single_k_vector_params(beam):
    g = beam.gain(k_vector[:, 0])
    assert g.shape == (num,)


@pytest.mark.parametrize("beam", models_vector, ids=beam_name)
def test_many_k_vector_params(beam):
    g = beam.gain(k_vector)
    assert g.shape == (num,)


@pytest.mark.parametrize("beam", models_scalar, ids=beam_name)
def test_single_k_scalar_params(beam):
    g = beam.gain(k_vector[:, 0])
    assert len(g.shape) == 0


@pytest.mark.parametrize("beam", models_scalar, ids=beam_name)
def test_many_k_scalar_params(beam):
    g = beam.gain(k_vector)
    assert g.shape == (num,)


if __name__ == "__main__":
    index = 2
    test_many_k_vector_params(models_vector[index])
    test_single_k_vector_params(models_vector[index])
    test_many_k_scalar_params(models_scalar[index])
    test_single_k_scalar_params(models_scalar[index])
