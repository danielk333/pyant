#!/usr/bin/env python

"""
Test Array
"""
import numpy as np
import scipy.constants as consts
import unittest

import pyant


class TestArray(unittest.TestCase):
    def make_test_beam(self):
        lam = 1.0
        antennas = np.zeros((3, 8, 1))
        antennas[0, :, 0] = np.arange(8) * lam / 2
        beam = pyant.models.Array(
            antennas=antennas,
        )
        param = pyant.models.ArrayParams(
            pointing=np.array([0, 0, 1], dtype=np.float64),
            frequency=consts.c / lam,
            polarization=beam.polarization.copy(),
        )

        return beam, param

    def test_peak_gain(self):
        beam, param = self.make_test_beam()
        peak_g_theorethical = 8
        peak_g = beam.gain(param.pointing, param)
        self.assertAlmostEqual(peak_g, peak_g_theorethical)
