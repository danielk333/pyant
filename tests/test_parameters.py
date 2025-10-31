#!/usr/bin/env python

"""
Test basic Beam utils
"""

from typing import ClassVar
from dataclasses import dataclass
import unittest
import numpy as np

import pyant

NUM = 100


@dataclass
class MockParam(pyant.types.Parameters):
    a: pyant.types.NDArray_N | float
    a_shape: ClassVar[None] = None
    b: pyant.types.NDArray_3xN | pyant.types.NDArray_3
    b_shape: ClassVar[tuple[int, ...]] = (3,)


class TestBeam(unittest.TestCase):
    def make_test_params_vector(self):
        param = MockParam(
            a=np.linspace(0, 1, NUM),
            b=np.ones((3, NUM), dtype=np.float64),
        )
        return param

    def make_test_params_scalar(self):
        param = MockParam(
            a=0.5,
            b=np.ones((3,), dtype=np.float64),
        )
        return param

    def test_init_scalar(self):
        param = self.make_test_params_scalar()
        del param

    def test_init_vector(self):
        param = self.make_test_params_vector()
        del param

    def test_keys(self):
        param = self.make_test_params_scalar()
        keys = param.keys
        for key in keys:
            assert hasattr(param, key + "_shape")

    def test_get_parameters_len_vector(self):
        param = self.make_test_params_vector()
        assert param.size() == NUM

    def test_get_parameters_len_scalar(self):
        param = self.make_test_params_scalar()
        assert param.size() is None

    def test_validate_shapes_var1(self):
        with self.assertRaises(pyant.types.SizeError):
            param = MockParam(
                a=0.5,
                b=np.ones((3, NUM), dtype=np.float64),
            )
            del param

    def test_validate_shapes_var2(self):
        with self.assertRaises(pyant.types.SizeError):
            param = MockParam(
                a=np.linspace(0, 1, NUM),
                b=np.ones((3,), dtype=np.float64),
            )
            del param

    def test_validate_shapes_var3(self):
        with self.assertRaises(pyant.types.SizeError):
            param = MockParam(
                a=np.linspace(0, 1, NUM + 1),
                b=np.ones((3, NUM), dtype=np.float64),
            )
            del param

    def test_validate_shapes_var4(self):
        with self.assertRaises(pyant.types.SizeError):
            param = MockParam(
                a=np.linspace(0, 1, NUM),
                b=np.ones((3, NUM + 1), dtype=np.float64),
            )
            del param
