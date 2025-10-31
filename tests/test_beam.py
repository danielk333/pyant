#!/usr/bin/env python

"""
Test basic Beam
"""

import unittest

import pyant


class MockBeam(pyant.Beam):
    def gain(self):
        pass


class TestBeam(unittest.TestCase):

    def make_test_beam(self):
        beam = MockBeam()
        return beam

    def test_init(self):
        beam = self.make_test_beam()
        beam.gain()
