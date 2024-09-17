#!/usr/bin/env python

"""Useful coordinate related functions.
"""

import numpy as np

CLOSE_TO_POLE_LIMIT = 1e-9**2
CLOSE_TO_POLE_LIMIT_rad = np.arctan(1 / np.sqrt(CLOSE_TO_POLE_LIMIT))


def cart_to_sph(vec, degrees=False):
    """Convert from Cartesian coordinates (east, north, up) to Spherical
    coordinates (azimuth, elevation, range) in a angle east of north and
    elevation fashion. Returns azimuth between [-pi, pi] and elevation between
    [-pi/2, pi/2].

    Parameters
    ----------
    vec : numpy.ndarray
        (3, N) or (3,) vector of Cartesian coordinates (east, north, up).
        This argument is vectorized in the second array dimension.
    degrees : bool
        If :code:`True`, use degrees. Else all angles are given in radians.

    Returns
    -------
    numpy.ndarray
        (3, N) or (3, ) vector of Spherical coordinates
        (azimuth, elevation, range).

    Notes
    -----
    Azimuth close to pole convention
        Uses a :code:`CLOSE_TO_POLE_LIMIT` constant when transforming determine
        if the point is close to the pole and sets the azimuth by definition
        to 0 "at" the poles for consistency.

    """

    r2_ = vec[0, ...] ** 2 + vec[1, ...] ** 2

    sph = np.empty(vec.shape, dtype=vec.dtype)

    if len(vec.shape) == 1:
        if r2_ < CLOSE_TO_POLE_LIMIT:
            sph[0] = 0.0
            sph[1] = np.sign(vec[2]) * np.pi * 0.5
        else:
            sph[0] = np.arctan2(vec[0], vec[1])
            sph[1] = np.arctan(vec[2] / np.sqrt(r2_))
    else:
        inds_ = r2_ < CLOSE_TO_POLE_LIMIT
        not_inds_ = np.logical_not(inds_)

        sph[0, inds_] = 0.0
        sph[1, inds_] = np.sign(vec[2, inds_]) * np.pi * 0.5
        sph[0, not_inds_] = np.arctan2(vec[0, not_inds_], vec[1, not_inds_])
        sph[1, not_inds_] = np.arctan(vec[2, not_inds_] / np.sqrt(r2_[not_inds_]))

    sph[2, ...] = np.sqrt(r2_ + vec[2, ...] ** 2)
    if degrees:
        sph[:2, ...] = np.degrees(sph[:2, ...])

    return sph


def sph_to_cart(vec, degrees=False):
    """Convert from spherical coordinates (azimuth, elevation, range) to
    Cartesian (east, north, up) in a angle east of north and elevation fashion.


    Parameters
    ----------
    vec : numpy.ndarray
        (3, N) or (3,) vector of Cartesian Spherical
        (azimuth, elevation, range).
        This argument is vectorized in the second array dimension.
    degrees : bool
        If :code:`True`, use degrees. Else all angles are given in radians.

    Returns
    -------
    numpy.ndarray
        (3, N) or (3, ) vector of Cartesian coordinates (east, north, up).

    """

    _az = vec[0, ...]
    _el = vec[1, ...]
    if degrees:
        _az, _el = np.radians(_az), np.radians(_el)
    cart = np.empty(vec.shape, dtype=vec.dtype)

    cart[0, ...] = vec[2, ...] * np.sin(_az) * np.cos(_el)
    cart[1, ...] = vec[2, ...] * np.cos(_az) * np.cos(_el)
    cart[2, ...] = vec[2, ...] * np.sin(_el)

    return cart


def vector_angle(a, b, degrees=False):
    """Angle between two vectors.

    Parameters
    ----------
    a : numpy.ndarray
        (3, N) or (3,) vector of Cartesian coordinates.
        This argument is vectorized in the second array dimension.
    b : numpy.ndarray
        (3, N) or (3,) vector of Cartesian coordinates.
        This argument is vectorized in the second array dimension.
    degrees : bool
        If :code:`True`, use degrees. Else all angles are given in radians.

    Returns
    -------
    numpy.ndarray or float
        (N, ) or float vector of angles between input vectors.

    Notes
    -----
    Definition
        :math:`\\theta = \\cos^{-1}\\frac{
            \\langle\\mathbf{a},\\mathbf{b}\\rangle
        }{
            |\\mathbf{a}||\\mathbf{b}|
        }`
        where :math:`\\langle\\mathbf{a},\\mathbf{b}\\rangle` is the dot
        product and :math:`|\\mathbf{a}|` denotes the norm.

    """
    a_norm = np.linalg.norm(a, axis=0)
    b_norm = np.linalg.norm(b, axis=0)

    if len(a.shape) == 1:
        proj = np.dot(a, b) / (a_norm * b_norm)
    elif len(b.shape) == 1:
        proj = np.dot(b, a) / (a_norm * b_norm)
    else:
        assert a.shape == b.shape, "Input shapes do not match"
        proj = np.sum(a * b, axis=0) / (a_norm * b_norm)

    if len(a.shape) == 1 and len(b.shape) == 1:
        if proj > 1.0:
            proj = 1.0
        elif proj < -1.0:
            proj = -1.0
    else:
        proj[proj > 1.0] = 1.0
        proj[proj < -1.0] = -1.0

    theta = np.arccos(proj)
    if degrees:
        theta = np.degrees(theta)

    return theta


def rot_mat_x(theta, dtype=np.float64, degrees=False):
    """Compute matrix for rotation of R3 vector through angle theta
    around the X-axis. For frame rotation, use the transpose.

    Parameters
    ----------
    theta : float
        Angle to rotate.
    dtype : numpy.dtype
        Numpy datatype of the rotation matrix.
    degrees : bool
        If :code:`True`, use degrees. Else all angles are given in radians.

    Returns
    -------
    numpy.ndarray
        (3, 3) Rotation matrix.

    """
    if degrees:
        theta = np.radians(theta)

    ca, sa = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]], dtype=dtype)


def rot_mat_y(theta, dtype=np.float64, degrees=False):
    """Compute matrix for rotation of R3 vector through angle theta
    around the Y-axis. For frame rotation, use the transpose.

    Parameters
    ----------
    theta : float
        Angle to rotate.
    dtype : numpy.dtype
        Numpy datatype of the rotation matrix.
    degrees : bool
        If :code:`True`, use degrees. Else all angles are given in radians.

    Returns
    -------
    numpy.ndarray
        (3, 3) Rotation matrix.

    """
    if degrees:
        theta = np.radians(theta)

    ca, sa = np.cos(theta), np.sin(theta)
    return np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]], dtype=dtype)


def rot_mat_z(theta, dtype=np.float64, degrees=False):
    """Compute matrix for rotation of R3 vector through angle theta
    around the Z-axis. For frame rotation, use the transpose.

    Parameters
    ----------
    theta : float
        Angle to rotate.
    dtype : numpy.dtype
        Numpy datatype of the rotation matrix.
    degrees : bool
        If :code:`True`, use degrees. Else all angles are given in radians.

    Returns
    -------
    numpy.ndarray
        (3, 3) Rotation matrix.

    """
    if degrees:
        theta = np.radians(theta)

    ca, sa = np.cos(theta), np.sin(theta)
    return np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)


def rot_mat_2d(theta, dtype=np.float64, degrees=True):
    """Matrix for rotation of R2 vector in the plane through angle theta
    For frame rotation, use the transpose.

    Parameters
    ----------
    theta : float
        Angle to rotate.
    dtype : numpy.dtype
        Numpy datatype of the rotation matrix.
    degrees : bool
        If :code:`True`, use degrees. Else all angles are given in radians.

    Returns
    -------
    numpy.ndarray
        (2, 2) Rotation matrix.

    """
    if degrees:
        theta = np.radians(theta)

    ca, sa = np.cos(theta), np.sin(theta)
    return np.array([[ca, -sa], [sa, ca]], dtype=dtype)


def scale_mat_2d(x, y):
    """Matrix for 2d scaling.

    Parameters
    ----------
    x : float
        Scaling coefficient for first coordinate axis.
    y : float
        Scaling coefficient for second coordinate axis.

    Returns
    -------
    numpy.ndarray
        (2, 2) Scaling matrix.
    """
    M_scale = np.zeros((2, 2), dtype=np.float64)
    M_scale[0, 0] = x
    M_scale[1, 1] = y
    return M_scale
