#!/usr/bin/env python

'''

'''

import numpy as np


def cart_to_sph(vec, radians=False):
    '''Convert from Cartesian coordinates (east, north, up) to Spherical coordinates (azimuth, elevation, range) in a degrees east of north and elevation fashion

    :param numpy.ndarray vec: Vector of Cartesian coordinates (east, north, up)
    :param bool radians: If `True` all angles returned are in radians instead of degrees
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
    :param bool radians: If `True` all angles returned are in radians instead of degrees
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
