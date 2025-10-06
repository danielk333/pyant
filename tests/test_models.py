#!/usr/bin/env python

"""
Test beam models
"""

import pytest
import numpy as np
import numpy.testing as nt

import pyant

num = 11
zenith = np.array([0, 0, 1], dtype=np.float64)
pointing = pyant.coordinates.sph_to_cart(np.array([0, 80.0, 1]), degrees=True)
k_vector = np.ones((3, num), dtype=np.float64)
k_vector[0, :] = 0
k_vector[1, :] = np.linspace(-0.1, 0.1, num)

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
]


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
