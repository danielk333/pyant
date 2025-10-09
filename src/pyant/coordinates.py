#!/usr/bin/env python

"""Useful coordinate related functions."""

import numpy as np

CLOSE_TO_POLE_LIMIT = 1e-9**2
CLOSE_TO_POLE_LIMIT_rad = np.arctan(1 / np.sqrt(CLOSE_TO_POLE_LIMIT))


def _clint(p, c, lim=1):
    """clip interval [p-c, p+c] to [-lim, lim] (lim=1 by default)"""
    return np.clip([p - c, p + c], -lim, lim)


def compute_j_grid(resolution):
    """Compute a grid of polarizations with given resolution
    """
    size = resolution**2
    thx = np.linspace(0, 2 * np.pi, num=resolution)
    thy = np.linspace(0, 2 * np.pi, num=resolution)

    jones_vecs = np.zeros((2, size), dtype=np.complex128)

    thxmat, thymat = np.meshgrid(thx, thy, sparse=False, indexing="ij")

    jones_vecs[0, :] = np.exp(1j * thxmat.reshape(1, size))
    jones_vecs[1, :] = np.exp(1j * thymat.reshape(1, size))

    return jones_vecs, thxmat, thymat


def compute_k_grid(pointing, resolution, centered, cmin):
    """Compute a grid of wave vector directions with given resolution
    """
    if centered:
        kx = np.linspace(*_clint(pointing[0], cmin), num=resolution)
        ky = np.linspace(*_clint(pointing[1], cmin), num=resolution)
    else:
        kx = np.linspace(-cmin, cmin, num=resolution)
        ky = np.linspace(-cmin, cmin, num=resolution)

    K = np.zeros((resolution, resolution, 2))

    # TODO: Refactor evaluation of function on a hemispherical domain to a function"
    K[:, :, 0], K[:, :, 1] = np.meshgrid(kx, ky, sparse=False, indexing="ij")
    size = resolution**2
    k = np.empty((3, size), dtype=np.float64)
    k[0, :] = K[:, :, 0].reshape(1, size)
    k[1, :] = K[:, :, 1].reshape(1, size)

    # circles in k space, centered on vertical and pointing, respectively
    z2 = k[0, :] ** 2 + k[1, :] ** 2
    z2_c = (pointing[0] - k[0, :]) ** 2 + (pointing[1] - k[1, :]) ** 2

    if centered:
        inds = np.logical_and(z2_c < cmin**2, z2 <= 1.0)
    else:
        inds = z2 < cmin**2
    not_inds = np.logical_not(inds)

    k[2, inds] = np.sqrt(1.0 - z2[inds])
    k[2, not_inds] = 0
    S = np.ones((size,)) * np.nan

    return S, K, k, inds, kx, ky


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
    theta : float or ndarray
        Angle to rotate.
    dtype : numpy.dtype
        Numpy datatype of the rotation matrix.
    degrees : bool
        If :code:`True`, use degrees. Else all angles are given in radians.

    Returns
    -------
    numpy.ndarray
        (3, 3) Rotation matrix, or (3, 3, n) tensor if theta is vector input.

    """
    if degrees:
        theta = np.radians(theta)
    if isinstance(theta, np.ndarray) and theta.ndim > 0:
        size = (3, 3, len(theta))
    else:
        size = (3, 3)

    ca, sa = np.cos(theta), np.sin(theta)
    rot = np.zeros(size, dtype=dtype)
    rot[0, 0, ...] = 1
    rot[1, 1, ...] = ca
    rot[1, 2, ...] = -sa
    rot[2, 1, ...] = sa
    rot[2, 2, ...] = ca
    return rot


def rot_mat_y(theta, dtype=np.float64, degrees=False):
    """Compute matrix for rotation of R3 vector through angle theta
    around the Y-axis. For frame rotation, use the transpose.

    Parameters
    ----------
    theta : float or ndarray
        Angle to rotate.
    dtype : numpy.dtype
        Numpy datatype of the rotation matrix.
    degrees : bool
        If :code:`True`, use degrees. Else all angles are given in radians.

    Returns
    -------
    numpy.ndarray
        (3, 3) Rotation matrix, or (3, 3, n) tensor if theta is vector input.

    """
    if degrees:
        theta = np.radians(theta)
    if isinstance(theta, np.ndarray) and theta.ndim > 0:
        size = (3, 3, len(theta))
    else:
        size = (3, 3)

    ca, sa = np.cos(theta), np.sin(theta)
    rot = np.zeros(size, dtype=dtype)
    rot[0, 0, ...] = ca
    rot[0, 2, ...] = sa
    rot[1, 1, ...] = 1
    rot[2, 0, ...] = -sa
    rot[2, 2, ...] = ca
    return rot


def rot_mat_z(theta, dtype=np.float64, degrees=False):
    """Compute matrix for rotation of R3 vector through angle theta
    around the Z-axis. For frame rotation, use the transpose.

    Parameters
    ----------
    theta : float or np.ndarray
        Angle to rotate.
    dtype : numpy.dtype
        Numpy datatype of the rotation matrix.
    degrees : bool
        If :code:`True`, use degrees. Else all angles are given in radians.

    Returns
    -------
    numpy.ndarray
        (3, 3) Rotation matrix, or (3, 3, n) tensor if theta is vector input.

    """
    if degrees:
        theta = np.radians(theta)
    if isinstance(theta, np.ndarray) and theta.ndim > 0:
        size = (3, 3, len(theta))
    else:
        size = (3, 3)

    ca, sa = np.cos(theta), np.sin(theta)
    rot = np.zeros(size, dtype=dtype)
    rot[0, 0, ...] = ca
    rot[0, 1, ...] = -sa
    rot[1, 0, ...] = sa
    rot[1, 1, ...] = ca
    rot[2, 2, ...] = 1
    return rot


def rot_mat_2d(theta, dtype=np.float64, degrees=False):
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
