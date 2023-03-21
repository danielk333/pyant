#!/usr/bin/env python

'''Useful coordinate related functions.

(c) 2020 Daniel Kastinen
'''

import numpy as np


def cart_to_sph(vec, radians=False):
    '''Convert from Cartesian coordinates (east, north, up) to Spherical coordinates (azimuth, elevation, range) in a degrees east of north and elevation fashion

    :param numpy.ndarray vec: Vector of Cartesian coordinates (east, north, up). This argument is vectorized in the second array dimension, i.e. it supports matrix `(3,n)` inputs as well as the standard `(3,)` vector inputs.
    :param bool radians: If :code:`True` all input/output angles are in radians, else they are in degrees
    :return: Vector of Spherical coordinates (azimuth, elevation, range)
    :rtype: numpy.ndarray
    '''

    r2_ = vec[0,...]**2 + vec[1,...]**2

    sph = np.empty(vec.shape, dtype=vec.dtype)

    if len(vec.shape) == 1:
        if r2_ < 1e-9**2:
            sph[0] = 0.0
            sph[1] = np.sign(vec[2])*np.pi*0.5
        else:
            sph[0] = np.pi/2 - np.arctan2(vec[1],vec[0])
            sph[1] = np.arctan(vec[2]/np.sqrt(r2_))
    else:
        inds_ = r2_ < 1e-9**2
        not_inds_ = np.logical_not(inds_)

        sph[0, inds_] = 0.0
        sph[1, inds_] = np.sign(vec[2,inds_])*np.pi*0.5
        sph[0, not_inds_] = np.pi/2 - np.arctan2(vec[1,not_inds_],vec[0,not_inds_])
        sph[1, not_inds_] = np.arctan(vec[2,not_inds_]/np.sqrt(r2_[not_inds_]))

    sph[2,...] = np.sqrt(r2_ + vec[2,...]**2)
    if not radians:
        sph[:2,...] = np.degrees(sph[:2,...])

    return sph


def sph_to_cart(vec, radians=False):
    '''Convert from spherical coordinates (azimuth, elevation, range) to Cartesian (east, north, up) in a degrees east of north and elevation fashion

    :param numpy.ndarray vec: Vector of Cartesian Spherical (azimuth, elevation, range). This argument is vectorized in the second array dimension, i.e. it supports matrix `(3,n)` inputs as well as the standard `(3,)` vector inputs.
    :param bool radians: If :code:`True` all input/output angles are in radians, else they are in degrees
    :return: Vector of Cartesian coordinates (east, north, up)
    :rtype: numpy.ndarray
    '''

    _az = vec[0,...]
    _el = vec[1,...]
    if not radians:
        _az, _el = np.radians(_az), np.radians(_el)
    cart = np.empty(vec.shape, dtype=vec.dtype)

    cart[0,...] = vec[2,...]*np.sin(_az)*np.cos(_el)
    cart[1,...] = vec[2,...]*np.cos(_az)*np.cos(_el)
    cart[2,...] = vec[2,...]*np.sin(_el)

    return cart



def vector_angle(a, b, radians=False):
    '''Angle in between two vectors :math:`\\theta = \\cos^{-1}\\frac{\\langle\\mathbf{a},\\mathbf{b}\\rangle}{|\\mathbf{a}||\\mathbf{b}|}`, where :math:`\\langle\\mathbf{a},\\mathbf{b}\\rangle` is the dot product and :math:`|\\mathbf{a}|` denotes the norm.
    
    :param numpy.ndarray a: Vector :math:`\\mathbf{a}`.
    :param numpy.ndarray b: Vector :math:`\\mathbf{b}`. This argument is vectorized in the second array dimension, i.e. it supports matrix `(3,n)` inputs as well as the standard `(3,)` vector inputs.
    :param bool radians: If :code:`True` all input/output angles are in radians, else they are in degrees
    :return: Angle :math:`\\theta` between vectors :math:`\\mathbf{a}` and :math:`\\mathbf{a}`.
    :rtype: float.
    '''
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b, axis=0)
    proj = np.dot(a,b)/(a_norm*b_norm)

    if len(b.shape) == 1:
        if proj > 1.0:
            proj = 1.0
        elif proj < -1.0:
            proj = -1.0
    else:
        proj[proj > 1.0] = 1.0
        proj[proj < -1.0] = -1.0

    theta = np.arccos(proj)
    if not radians:
        theta = np.degrees(theta)

    return theta


def rot_mat_z(theta, dtype=np.float64, radians=False):
    '''Compute matrix for rotation of R3 vector through angle theta
    around the Z-axis.  For frame rotation, use the transpose.
    
    :param float theta: Angle to rotate.
    :param numpy.dtype dtype: The data-type of the output matrix.
    :param bool radians: Uses radians if set to :code:`True`.

    :return: Rotation matrix
    :rtype: (3,3) numpy.ndarray
    '''
    if not radians:
        theta = np.radians(theta)

    ca, sa = np.cos(theta), np.sin(theta)
    return np.array([[ca, -sa, 0.],
                     [sa,  ca, 0.],
                     [0.,  0., 1.]], dtype=dtype)


def rot_mat_x(theta, dtype=np.float64, radians=False):
    '''Compute matrix for rotation of R3 vector through angle theta
    around the X-axis.  For frame rotation, use the transpose.

    
    :param float theta: Angle to rotate.
    :param numpy.dtype dtype: The data-type of the output matrix.
    :param bool radians: Uses radians if set to :code:`True`.

    :return: Rotation matrix
    :rtype: (3,3) numpy.ndarray
    '''
    if not radians:
        theta = np.radians(theta)

    ca, sa = np.cos(theta), np.sin(theta)
    return np.array([[1., 0.,  0.],
                     [0., ca, -sa],
                     [0., sa,  ca]], dtype=dtype)


def rot_mat_y(theta, dtype=np.float64, radians=False):
    '''Compute matrix for rotation of R3 vector through angle theta
    around the Y-axis.  For frame rotation, use the transpose.
    
    :param float theta: Angle to rotate.
    :param numpy.dtype dtype: The data-type of the output matrix.
    :param bool radians: Uses radians if set to :code:`True`.

    :return: Rotation matrix
    :rtype: (3,3) numpy.ndarray
    '''
    if not radians:
        theta = np.radians(theta)

    ca, sa = np.cos(theta), np.sin(theta)
    return np.array([[ ca, 0., sa],
                     [ 0., 1., 0.],
                     [-sa, 0., ca]], dtype=dtype)


def rot2d(theta, dtype=np.float64, radians=True):
    '''Matrix for rotation of R2 vector in the plane through angle theta
    For frame rotation, use the transpose.

    :param float theta: Angle to rotate.
    :param numpy.dtype dtype: The data-type of the output matrix.
    :param bool radians: Uses radians unless set to :code:`False`.

    :return: Rotation matrix
    :rtype: (2,2) numpy.ndarray
    '''
    if not radians:
        theta = np.radians(theta)

    ca, sa = np.cos(theta), np.sin(theta)
    return np.array([[ca, -sa],
                     [sa,  ca]], dtype=dtype)


def scale2d(x,y):
    '''Matrix for scaling in the plane

    #TODO docstring
    '''
    M_rot = np.zeros((2,2), dtype=np.float64)
    M_rot[0,0] = x
    M_rot[1,1] = y
    return M_rot
