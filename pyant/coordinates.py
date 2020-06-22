#!/usr/bin/env python

'''Useful coordinate related functions.

(c) 2020 Daniel Kastinen
'''

import numpy as np


def cart_to_sph(vec, radians=False):
    '''Convert from Cartesian coordinates (east, north, up) to Spherical coordinates (azimuth, elevation, range) in a degrees east of north and elevation fashion

    :param numpy.ndarray vec: Vector of Cartesian coordinates (east, north, up)
    :param bool radians: If :code:`True` all input/output angles are in radians, else they are in degrees
    :return: Vector of Spherical coordinates (azimuth, elevation, range)
    :rtype: numpy.ndarray
    '''
    x = vec[0]
    y = vec[1]
    z = vec[2]
    r_ = np.sqrt(x**2 + y**2)
    if r_ < 1e-9:
        el = np.sign(z)*np.pi*0.5
        az = 0.0
    else:
        el = np.arctan(z/r_)
        az = np.pi/2 - np.arctan2(y,x)
    r = np.sqrt(x**2 + y**2 +z**2)
    if radians:
        return np.array([az, el, r])
    else:
        return np.array([np.degrees(az), np.degrees(el), r])


def sph_to_cart(vec, radians=False):
    '''Convert from spherical coordinates (azimuth, elevation, range) to Cartesian (east, north, up) in a degrees east of north and elevation fashion

    :param numpy.ndarray vec: Vector of Cartesian Spherical (azimuth, elevation, range)
    :param bool radians: If :code:`True` all input/output angles are in radians, else they are in degrees
    :return: Vector of Cartesian coordinates (east, north, up)
    :rtype: numpy.ndarray
    '''
    _az = vec[0]
    _el = vec[1]
    if not radians:
        _az, _el = np.radians(_az), np.radians(_el)
    return vec[2]*np.array([np.sin(_az)*np.cos(_el), np.cos(_az)*np.cos(_el), np.sin(_el)])



def vector_angle(a, b, radians=False):
    '''Angle in between two vectors :math:`\\theta = \\cos^{-1}\\frac{\\langle\\mathbf{a},\\mathbf{b}\\rangle}{|\\mathbf{a}||\\mathbf{b}|}`, where :math:`\\langle\\mathbf{a},\\mathbf{b}\\rangle` is the dot product and :math:`|\\mathbf{a}|` denotes the norm.
    
    :param numpy.ndarray a: Vector :math:`\\mathbf{a}`.
    :param numpy.ndarray b: Vector :math:`\\mathbf{b}`.
    :param bool radians: If :code:`True` all input/output angles are in radians, else they are in degrees
    :return: Angle :math:`\\theta` between vectors :math:`\\mathbf{a}` and :math:`\\mathbf{a}`.
    :rtype: float.
    '''
    proj = np.dot(a,b)/(np.sqrt(np.dot(a,a))*np.sqrt(np.dot(b,b)))
    if proj > 1.0:
        proj = 1.0
    elif proj < -1.0:
        proj = -1.0
    theta = np.arccos(proj)
    if not radians:
        theta = np.degrees(theta)
    return theta


def rot_mat_z(theta, dtype=np.float, radians=False):
    '''Generates the 3D transformation matrix for rotation around Z-axis.
    
    :param float theta: Angle to rotate.
    :param numpy.dtype dtype: The data-type of the output matrix.
    :param bool radians: Uses radians if set to :code:`True`.

    :return: Rotation matrix
    :rtype: (3,3) numpy.ndarray
    '''
    if not radians:
        theta = np.radians(theta)

    R = np.zeros((3,3), dtype=dtype)
    R[0,0] = np.cos(theta)
    R[0,1] = -np.sin(theta)
    R[1,0] = np.sin(theta)
    R[1,1] = np.cos(theta)
    R[2,2] = 1.0
    return R


def rot_mat_x(theta, dtype=np.float, radians=False):
    '''Generates the 3D transformation matrix for rotation around X-axis.
    
    :param float theta: Angle to rotate.
    :param numpy.dtype dtype: The data-type of the output matrix.
    :param bool radians: Uses radians if set to :code:`True`.

    :return: Rotation matrix
    :rtype: (3,3) numpy.ndarray
    '''
    if not radians:
        theta = np.radians(theta)

    R = np.zeros((3,3), dtype=dtype)
    R[1,1] = np.cos(theta)
    R[1,2] = -np.sin(theta)
    R[2,1] = np.sin(theta)
    R[2,2] = np.cos(theta)
    R[0,0] = 1.0
    return R


def rot_mat_y(theta, dtype=np.float, radians=False):
    '''Generates the 3D transformation matrix for rotation around Y-axis.
    
    :param float theta: Angle to rotate.
    :param numpy.dtype dtype: The data-type of the output matrix.
    :param bool radians: Uses radians if set to :code:`True`.

    :return: Rotation matrix
    :rtype: (3,3) numpy.ndarray
    '''
    if not radians:
        theta = np.radians(theta)

    R = np.zeros((3,3), dtype=dtype)
    R[0,0] = np.cos(theta)
    R[0,2] = np.sin(theta)
    R[2,0] = -np.sin(theta)
    R[2,2] = np.cos(theta)
    R[1,1] = 1.0
    return R

def rot2d(theta):
    '''Matrix for rotation in the plane

    #TODO docstring
    '''
    M_rot = np.empty((2,2), dtype=np.float)
    M_rot[0,0] = np.cos(theta)
    M_rot[1,0] = np.sin(theta)
    M_rot[0,1] = -np.sin(theta)
    M_rot[1,1] = np.cos(theta)
    return M_rot

def scale2d(x,y):
    '''Matrix for scaling in the plane

    #TODO docstring
    '''
    M_rot = np.zeros((2,2), dtype=np.float)
    M_rot[0,0] = x
    M_rot[1,1] = y
    return M_rot
