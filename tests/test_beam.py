#!/usr/bin/env python

"""
Test basic Beam utils
"""

import unittest
import scipy.constants
import numpy as np
import numpy.testing as nt

import pyant


class MockBeam(pyant.Beam):
    def gain(self):
        pass


class TestBeam(unittest.TestCase):
    def setUp(self):
        self.data = {
            "azimuth": np.array([0, 90, 180], dtype=np.float64),
            "elevation": np.array([90, 0, 0], dtype=np.float64),
        }
        self.data["k_all"] = pyant.coordinates.sph_to_cart(
            np.array(
                [
                    self.data["azimuth"],
                    self.data["elevation"],
                    [1, 1, 1],
                ]
            ),
            degrees=True,
        )
        self.data["k"] = self.data["k_all"][:, :2]
        self.data["param_sizes"] = {
            "a": 0,
            "b": 2,
            "c": 0,
            "d": 2,
        }
        self.data["vec_size"] = 2
        self.data["frequency"] = 1e8
        self.data["wavelength"] = scipy.constants.c / 1e8

    def make_test_beam(self):
        beam = MockBeam()
        return beam

    def make_test_beam_params_variaty(self):
        beam = MockBeam()
        beam.parameters["a"] = 0
        beam.parameters["b"] = np.arange(2)
        beam.parameters["c"] = np.arange(2)
        beam.parameters["d"] = np.ones((2, 2))
        beam.parameters_shape["c"] = (2,)
        beam.parameters_shape["d"] = (2,)
        return beam

    def make_test_beam_params_vector(self):
        beam = MockBeam()
        beam.parameters["c"] = np.arange(2)
        beam.parameters["d"] = np.ones((2, 2))
        beam.parameters_shape["d"] = (2,)
        return beam

    def make_test_beam_params_scalar(self):
        beam = MockBeam()
        beam.parameters["frequency"] = self.data["frequency"]
        return beam

    def make_test_beam_params_point(self):
        beam = MockBeam()
        beam.parameters["pointing"] = self.data["k"][:, 0]
        return beam

    def test_init(self):
        beam = self.make_test_beam()
        beam.gain()

    def test_keys(self):
        beam = self.make_test_beam_params_variaty()
        keys = beam.keys
        for key in self.data["param_sizes"]:
            assert key in keys
        for key in keys:
            assert key in self.data["param_sizes"]

    def test_get_parameters_len(self):
        beam = self.make_test_beam_params_variaty()
        for key in beam.keys:
            assert self.data["param_sizes"][key] == beam._get_parameter_len(key), key

    def test_size(self):
        beam = self.make_test_beam_params_vector()
        self.assertEqual(beam.size, self.data["vec_size"])
        beam = self.make_test_beam_params_scalar()
        self.assertEqual(beam.size, 0)

    def test_param_validator(self):
        beam = self.make_test_beam_params_vector()
        beam.validate_parameter_shapes()
        beam = self.make_test_beam_params_variaty()
        with self.assertRaises(AssertionError):
            beam.validate_parameter_shapes()

    def test_k_validator(self):
        beam = self.make_test_beam_params_vector()
        beam.validate_k_shape(self.data["k"])
        beam.validate_k_shape(self.data["k"][:, 0])
        with self.assertRaises(AssertionError):
            beam.validate_k_shape(self.data["k_all"])

    def test_freq_lam(self):
        beam = self.make_test_beam_params_scalar()
        self.assertAlmostEqual(beam.frequency, self.data["frequency"])
        self.assertAlmostEqual(beam.wavelength, self.data["wavelength"])
        beam.frequency = 1e7
        newlam = scipy.constants.c / beam.frequency
        self.assertAlmostEqual(beam.frequency, 1e7)
        self.assertAlmostEqual(beam.wavelength, newlam)
        newlam = scipy.constants.c / 5e7
        beam.wavelength = newlam
        self.assertAlmostEqual(beam.frequency, 5e7)
        self.assertAlmostEqual(beam.wavelength, newlam)

    def test_angle(self):
        beam = self.make_test_beam_params_point()
        k = np.array([1, 0, 0], dtype=np.float64)
        th = beam.angle(k, degrees=True)
        nt.assert_almost_equal(th, 90.0)

        th = beam.angle(k, degrees=False)
        nt.assert_almost_equal(th, np.pi / 2)

        th = beam.sph_angle(90, 0, degrees=True)
        nt.assert_almost_equal(th, 90.0)

    def test_azel_to_numpy(self):
        azelr = pyant.Beam._azel_to_numpy(self.data["azimuth"], self.data["elevation"])
        nt.assert_equal(azelr[0, :], self.data["azimuth"])
        nt.assert_equal(azelr[1, :], self.data["elevation"])
        nt.assert_equal(azelr[2, :], np.ones_like(self.data["azimuth"]))

        azelr = pyant.Beam._azel_to_numpy(0, self.data["elevation"])
        nt.assert_equal(azelr[0, :], 0)
        nt.assert_equal(azelr[1, :], self.data["elevation"])
        nt.assert_equal(azelr[2, :], np.ones_like(self.data["azimuth"]))

        azelr = pyant.Beam._azel_to_numpy(self.data["azimuth"], 0)
        nt.assert_equal(azelr[0, :], self.data["azimuth"])
        nt.assert_equal(azelr[1, :], 0)
        nt.assert_equal(azelr[2, :], np.ones_like(self.data["azimuth"]))

    def test_sph_point(self):
        # TODO: implement these
        pass
