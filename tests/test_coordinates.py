#!/usr/bin/env python

'''
Test basic kepler functions
'''

import unittest
import numpy as np
import numpy.testing as nt
from numpy import pi

import pyant.coordinates as coord


class TestCartSph(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
            [1, 1, 1],
            [0, 1, 1],
        ], dtype=np.float64)
        self.X = self.X.T
        self.Y = np.array([
            [pi/2, 0, 1],
            [0, 0, 1],
            [-pi/2, 0, 1],
            [pi, 0, 1],
            [0, pi/2, 1],
            [0, -pi/2, 1],
            [pi/4, np.arccos(np.sqrt(2/3)), np.sqrt(3)],
            [0, pi/4, np.sqrt(2)],
        ], dtype=np.float64)
        self.Y = self.Y.T
        self.num = self.X.shape[1]

        self.Yd = self.Y.copy()
        self.Yd[:2, :] = np.degrees(self.Yd[:2, :])

    def test_cart_to_sph(self):
        for ind in range(self.num):
            y = coord.cart_to_sph(self.X[:, ind], degrees=False)
            print(f'T({self.X[:, ind]}) = {y} == {self.Y[:, ind]}')
            nt.assert_array_almost_equal(self.Y[:, ind], y)

    def test_sph_to_cart(self):
        for ind in range(self.num):
            x = coord.sph_to_cart(self.Y[:, ind], degrees=False)
            print(f'T^-1({self.Y[:, ind]}) = {x} == {self.X[:, ind]}')
            nt.assert_array_almost_equal(self.X[:, ind], x)

    def test_cart_to_sph_vectorized(self):
        Y = coord.cart_to_sph(self.X, degrees=False)
        nt.assert_array_almost_equal(self.Y, Y)

    def test_sph_to_cart_vectorized(self):
        X = coord.sph_to_cart(self.Y, degrees=False)
        nt.assert_array_almost_equal(self.X, X)

    def test_degrees_keyword(self):
        Yd = coord.cart_to_sph(self.X, degrees=True)
        nt.assert_array_almost_equal(self.Yd, Yd)

        X = coord.sph_to_cart(self.Yd, degrees=True)
        nt.assert_array_almost_equal(self.X, X)

    def test_inverse_consistency(self):
        num = 100
        [az, el] = np.meshgrid(
            np.linspace(-pi, pi, num),
            np.linspace(-pi/2, pi/2, num),
        )
        az = az.flatten()
        el = el.flatten()

        # By convention, azimuth information is lost at pole
        az[np.abs(el) > coord.CLOSE_TO_POLE_LIMIT_rad] = 0

        vec = np.ones((3,), dtype=np.float64)
        for ind in range(num**2):
            vec[0] = az[ind]
            vec[1] = el[ind]
            cart = coord.sph_to_cart(vec, degrees=False)
            ang = coord.cart_to_sph(cart, degrees=False)
            nt.assert_array_almost_equal(vec, ang)

    def test_inverse_edge_cases(self):
        Y = np.array([
            [np.pi, 0, 1],
            [-np.pi, 0, 1],
        ], dtype=np.float64).T

        for ind in range(Y.shape[1]):
            X = coord.sph_to_cart(Y[:, ind], degrees=False)
            Yp = coord.cart_to_sph(X, degrees=False)
            nt.assert_array_almost_equal(Yp, Y[:, ind])


class TestAngles(unittest.TestCase):

    def setUp(self):
        self.A = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 1, 0],
        ], dtype=np.float64)
        self.A = self.A.T
        self.B = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 0, -1],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float64)
        self.B = self.B.T
        self.theta = np.array([
            0,
            pi/2,
            pi/2,
            pi,
            pi,
            pi/4,
            pi/4,
        ], dtype=np.float64)
        self.p = np.array([0, 1, 0], dtype=np.float64)
        self.P = np.array([
            [1, 0, 0],
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
        ], dtype=np.float64).T
        self.phi = np.array([
            pi/2,
            pi/4,
            pi/2,
            pi/4,
        ], dtype=np.float64)

    def test_vector_angle(self):
        for ind in range(len(self.theta)):
            th = coord.vector_angle(
                self.A[:, ind],
                self.B[:, ind],
                degrees=False,
            )
            nt.assert_almost_equal(th, self.theta[ind])

    def test_vector_angle_first_vectorized(self):
        th = coord.vector_angle(self.P, self.p, degrees=False)
        nt.assert_array_almost_equal(th, self.phi)

    def test_vector_angle_second_vectorized(self):
        th = coord.vector_angle(self.p, self.P, degrees=False)
        nt.assert_array_almost_equal(th, self.phi)

    def test_vector_angle_both_vectorized(self):
        th = coord.vector_angle(self.A, self.B, degrees=False)
        nt.assert_array_almost_equal(th, self.theta)


class TestRotations(unittest.TestCase):

    def setUp(self):
        self.basis = np.eye(3, dtype=np.float64)
        self.basis_x_90deg = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ], dtype=np.float64)
        self.basis_y_90deg = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0],
        ], dtype=np.float64)
        self.basis_z_90deg = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float64)

    def test_rot_mat_x(self):
        R = coord.rot_mat_x(pi/2)
        r_basis = R @ self.basis
        nt.assert_array_almost_equal(r_basis, self.basis_x_90deg)
        basis = R.T @ r_basis
        nt.assert_array_almost_equal(basis, self.basis)

    def test_rot_mat_y(self):
        R = coord.rot_mat_y(pi/2)
        r_basis = R @ self.basis
        nt.assert_array_almost_equal(r_basis, self.basis_y_90deg)
        basis = R.T @ r_basis
        nt.assert_array_almost_equal(basis, self.basis)

    def test_rot_mat_z(self):
        R = coord.rot_mat_z(pi/2)
        r_basis = R @ self.basis
        nt.assert_array_almost_equal(r_basis, self.basis_z_90deg)
        basis = R.T @ r_basis
        nt.assert_array_almost_equal(basis, self.basis)


class TestScale(unittest.TestCase):

    def test_scale_mat_2d_x(self):
        a = np.array([1, 0], dtype=np.float64)
        x_scale = 3
        b = a.copy()
        b[0] *= x_scale
        M = coord.scale_mat_2d(x_scale, 1)

        bp = M @ a
        nt.assert_array_almost_equal(bp, b)

    def test_scale_mat_2d_y(self):
        a = np.array([0, 1], dtype=np.float64)
        y_scale = 3
        b = a.copy()
        b[1] *= y_scale
        M = coord.scale_mat_2d(1, y_scale)

        bp = M @ a
        nt.assert_array_almost_equal(bp, b)

    def test_scale_mat_2d_xy(self):
        a = np.array([1, 1], dtype=np.float64)
        x_scale = 3
        y_scale = 2
        b = a.copy()
        b[0] *= x_scale
        b[1] *= y_scale
        M = coord.scale_mat_2d(x_scale, y_scale)

        bp = M @ a
        nt.assert_array_almost_equal(bp, b)
