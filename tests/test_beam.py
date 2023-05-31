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

        class TestBeam(pyant.Beam):
            def gain(self):
                pass

            def sph_gain(self):
                pass

        self.beam1 = TestBeam(
            azimuth=self.data["azimuth"],
            elevation=self.data["elevation"],
            frequency=self.data["frequency"],
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
        )

    def test_get_parameters(self):
        data = self.beam1.get_parameters()
        nt.assert_array_almost_equal(
            self.data["frequency"],
            data[self.beam1._inds["frequency"]],
        )

        data = self.beam2.get_parameters()
        nt.assert_array_almost_equal(
            self.data["radius"],
            data[self.beam2._inds["radius"]],
        )

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
