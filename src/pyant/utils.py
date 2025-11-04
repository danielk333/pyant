import numpy as np
import spacecoords.spherical as sph
import spacecoords.linalg as linalg
from .beam import get_and_validate_k_shape
from .types import NDArray_3, NDArray_3xN, NDArray_N, P, ParamError


def local_to_pointing(
    k: NDArray_3xN | NDArray_3,
    parameters: P,
) -> tuple[NDArray_N | float, NDArray_N | float]:
    """Convert from local wave vector direction to bore-sight relative
    longitudinal and transverse angles.
    """
    if "pointing" not in parameters.keys:
        raise ParamError("Can only transform to boresight coordiantes if parameters has pointing")

    size = parameters.size()
    k_len = get_and_validate_k_shape(size, k)
    if k_len == 0:
        return np.empty((0,), dtype=k.dtype), np.empty((0,), dtype=k.dtype)

    azelr = sph.cart_to_sph(parameters.pointing, degrees=False)  # type: ignore

    k = k / np.linalg.norm(k, axis=0)

    Rz = linalg.rot_mat_z(azelr[0, ...], degrees=False)
    Rx = linalg.rot_mat_x(np.pi / 2 - azelr[1, ...], degrees=False)

    # Look direction rotated into the radar's boresight system
    if size is not None and k_len is not None:
        kb = np.einsum("ijk,jk->ik", Rx, np.einsum("ijk,jk->ik", Rz, k))
    elif size is not None and k_len is None:
        kb = np.einsum(
            "ijk,jk->ik",
            Rx,
            np.einsum("ijk,jk->ik", Rz, np.broadcast_to(k.reshape(3, 1), (3, size))),
        )
    elif size is None:
        kb = Rx @ Rz @ k

    # angle of kb from y;z plane, clock wise
    # ( https://www.cv.nrao.edu/~sransom/web/Ch3.html )
    phi = np.arcsin(kb[0, ...])  # Angle of look to left (-) or right (+) of b.s.

    # angle of kb from x;z plane, counter-clock wise
    # ( https://www.cv.nrao.edu/~sransom/web/Ch3.html )
    theta = np.arcsin(kb[1, ...])  # Angle of look above (-) or below (+) boresight

    return theta, phi


def pointing_to_local(
    theta: NDArray_N | float,
    phi: NDArray_N | float,
    parameters: P,
) -> NDArray_3xN | NDArray_3:
    """Convert from bore-sight relative longitudinal and transverse angles
    to local wave vector direction.
    """
    if not hasattr(parameters, "parameters"):
        raise ParamError("Can only transform to boresight coordiantes if parameters has pointing")

    size = parameters.size()
    azelr = sph.cart_to_sph(parameters.pointing, degrees=False)  # type: ignore

    sz: tuple[int, ...] = (3,)
    k_len = 0
    if isinstance(theta, np.ndarray) and theta.ndim > 0:
        sz = sz + (len(theta),)
        k_len = len(theta)
    elif isinstance(phi, np.ndarray) and phi.ndim > 0:
        sz = sz + (len(phi),)
        k_len = len(phi)

    kb = np.zeros(sz, dtype=np.float64)
    kb[0, ...] = np.sin(phi)
    kb[1, ...] = np.sin(theta)
    kb[2, ...] = np.sqrt(1 - kb[0, ...] ** 2 - kb[1, ...] ** 2)

    Rz = linalg.rot_mat_z(azelr[0, ...], degrees=False)
    Rx = linalg.rot_mat_x(np.pi / 2 - azelr[1, ...], degrees=False)

    # Look direction rotated from the radar's boresight system
    if size is not None and k_len is not None:
        k = np.einsum(
            "ijk,jk->ik",
            np.einsum("ijk->jik", Rx),
            np.einsum("ijk,jk->ik", np.einsum("ijk->jik", Rz), kb),
        )
    elif size is not None and k_len is None:
        k = np.einsum(
            "ijk,jk->ik",
            np.einsum("ijk->jik", Rx),
            np.einsum(
                "ijk,jk->ik",
                np.einsum("ijk->jik", Rz),
                np.broadcast_to(kb.reshape(3, 1), (3, size)),
            ),
        )
    elif size is None:
        k = Rz.T @ Rx.T @ kb

    return k


def _clint(p: float, c: float, lim: float = 1) -> tuple[float, float]:
    """clip interval [p-c, p+c] to [-lim, lim] (lim=1 by default)"""
    x = np.clip([p - c, p + c], -lim, lim)
    return x[0], x[1]


def compute_j_grid(resolution: int):
    """Compute a grid of polarizations with given resolution"""
    size = resolution**2
    thx = np.linspace(0, 2 * np.pi, num=resolution)
    thy = np.linspace(0, 2 * np.pi, num=resolution)

    jones_vecs = np.zeros((2, size), dtype=np.complex128)

    thxmat, thymat = np.meshgrid(thx, thy, sparse=False, indexing="ij")

    jones_vecs[0, :] = np.exp(1j * thxmat.reshape(1, size))
    jones_vecs[1, :] = np.exp(1j * thymat.reshape(1, size))

    return jones_vecs, thxmat, thymat


def compute_k_grid(pointing: NDArray_3, resolution: int, centered: bool, cmin: float):
    """Compute a grid of wave vector directions with given resolution"""
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
