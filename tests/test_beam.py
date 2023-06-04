#!/usr/bin/env python

"""
Test basic kepler functions
"""

import unittest
import numpy as np
import numpy.testing as nt

import pyant


class TestBeam(unittest.TestCase):
    def setUp(self):
        self.data = {
            "azimuth": [0, 45.0, 0],
            "elevation": [90.0, 80.0, 60.0],
            "frequency": [930e6, 230e6],
            "radius": 23.0,
        }
        self.k = pyant.coordinates.sph_to_cart(
            np.array(
                [
                    self.data["azimuth"],
                    self.data["elevation"],
                    [1, 1, 1],
                ]
            ),
            degrees=True,
        )

        class TestBeam(pyant.Beam):
            def gain(self):
                pass

            def sph_gain(self):
                pass

        self.beam1 = TestBeam(
            azimuth=self.data["azimuth"],
            elevation=self.data["elevation"],
            frequency=self.data["frequency"],
            degrees=True,
        )

        class TestBeam2(pyant.Beam):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.register_parameter("radius")
                self.fill_parameter("radius", kwargs["radius"])

            def gain(self):
                pass

            def sph_gain(self):
                pass

        self.beam2 = TestBeam2(
            azimuth=self.data["azimuth"],
            elevation=self.data["elevation"],
            frequency=self.data["frequency"],
            radius=self.data["radius"],
            degrees=True,
        )

    def test_get_parameters(self):
        params, shape = self.beam1.get_parameters()
        nt.assert_array_almost_equal(
            self.data["frequency"],
            params[self.beam1._inds["frequency"]],
        )

        params, shape = self.beam2.get_parameters()
        nt.assert_array_almost_equal(
            self.data["radius"],
            params[self.beam2._inds["radius"]],
        )

    def test_get_parameters_ind(self):
        params, shape = self.beam1.get_parameters(ind=(0, 0))
        nt.assert_array_almost_equal(params[0], self.k[:, 0])
        nt.assert_almost_equal(params[1], self.data["frequency"][0])

        params, shape = self.beam1.get_parameters(ind=(0, 0), named=True)
        nt.assert_array_almost_equal(params["pointing"], self.k[:, 0])
        nt.assert_almost_equal(params["frequency"], self.data["frequency"][0])

    def test_broadcast_params(self):
        base_params, shape = self.beam1.get_parameters(named=True)
        with self.assertRaises(AssertionError):
            params, G = self.beam1.broadcast_params(base_params, shape, 0)
        base_params, shape = self.beam1.get_parameters(ind=(slice(None), 0), named=True)

        params, G = self.beam1.broadcast_params(base_params, shape, 0)
        assert G.shape == (len(self.data["azimuth"]),)

        params, G = self.beam1.broadcast_params(base_params, shape, 10)
        assert G.shape == (10, len(self.data["azimuth"]))

        base_params, shape = self.beam1.get_parameters(ind=(0, 0), named=True)
        params, G = self.beam1.broadcast_params(base_params, shape, 0)
        assert len(G.shape) == 0

        params, G = self.beam1.broadcast_params(base_params, shape, 10)
        assert G.shape == (10,)

        base_params, shape = self.beam1.get_parameters(ind=(0, slice(None)), named=True)
        params, G = self.beam1.broadcast_params(base_params, shape, 0)
        assert G.shape == (len(self.data["frequency"]),)

        params, G = self.beam1.broadcast_params(base_params, shape, 10)
        assert G.shape == (10, len(self.data["frequency"]))

        params, G = self.beam1.broadcast_params(base_params, shape, 0)
        for key in params:
            if key == "pointing":
                assert params[key].shape == (3,)
            else:
                assert params[key].shape == G.shape

        params, G = self.beam1.broadcast_params(base_params, shape, 10)
        for key in params:
            if key == "pointing":
                assert params[key].shape == (3,)
            else:
                assert params[key].shape == G.shape

    def test_set_parameters_property(self):
        f = np.array([2.0])
        self.beam1.frequency = f
        nt.assert_array_almost_equal(
            f,
            self.beam1.frequency,
        )

        r = np.array([1.0, 2.0])
        self.beam2.parameters["radius"] = r
        nt.assert_array_almost_equal(
            r,
            self.beam2.parameters["radius"],
        )
        rp = 3.0
        self.beam2.fill_parameter("radius", rp)
        nt.assert_array_almost_equal(
            r * 0 + rp,
            self.beam2.parameters["radius"],
        )

    def test_wavelength(self):
        self.beam1.frequency = [1]
        c = self.beam1.wavelength
        self.beam1.wavelength = 1
        nt.assert_almost_equal(self.beam1.frequency[0], c)

    def test_get_pointing(self):
        nt.assert_array_almost_equal(self.beam1.pointing, self.k)
        nt.assert_array_almost_equal(self.beam1.azimuth, self.data["azimuth"])
        nt.assert_array_almost_equal(self.beam1.elevation, self.data["elevation"])

    def test_set_sph_pointing(self):
        k = np.array([1, 0, 0], dtype=np.float64)
        k.shape = (3, 1)
        az = np.array([90.0])
        el = np.array([0.0])

        self.beam1.sph_point(az, el, degrees=True)
        nt.assert_array_almost_equal(self.beam1.pointing, k)
        nt.assert_array_almost_equal(self.beam1.azimuth, az)
        nt.assert_array_almost_equal(self.beam1.elevation, el)

        self.beam1.sph_point(self.data["azimuth"], self.data["elevation"], degrees=True)
        nt.assert_array_almost_equal(self.beam1.pointing, self.k)
        nt.assert_array_almost_equal(self.beam1.azimuth, self.data["azimuth"])
        nt.assert_array_almost_equal(self.beam1.elevation, self.data["elevation"])

        self.beam1.point(k)
        nt.assert_array_almost_equal(self.beam1.pointing, k)
        nt.assert_array_almost_equal(self.beam1.azimuth, az)
        nt.assert_array_almost_equal(self.beam1.elevation, el)

    def test_angle(self):
        k = np.array([1, 0, 0], dtype=np.float64)
        th = self.beam1.angle(k, degrees=True)
        nt.assert_almost_equal(th[0], 90.0)

        th = self.beam1.angle(k, degrees=False)
        nt.assert_almost_equal(th[0], np.pi / 2)

        th = self.beam1.sph_angle(90, 0, degrees=True)
        nt.assert_almost_equal(th[0], 90.0)

    def test_get_parameter_len(self):
        assert self.beam1._get_parameter_len("frequency") == 2

    def test_shape(self):
        shape = self.beam2.named_shape
        assert list(shape.keys()) == ["pointing", "frequency", "radius"]
        assert list(shape.values()) == [3, 2, 1]
        assert self.beam2.shape == (3, 2, 1)

    def test_keys(self):
        assert self.beam2.keys == tuple(self.beam2.parameters.keys())

    def test_ind_to_dict(self):
        inds = self.beam1.ind_to_dict(None)
        assert inds["pointing"] == (slice(None), slice(None))
        assert inds["frequency"] == (slice(None),)

        inds = self.beam1.ind_to_dict({"pointing": 0})
        assert inds["pointing"] == (slice(None), 0)
        assert inds["frequency"] == (slice(None),)

        inds = self.beam1.ind_to_dict((slice(1, None), 1))
        assert inds["pointing"] == (slice(None), slice(1, None))
        assert inds["frequency"] == (1,)
