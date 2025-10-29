import numpy as np


def _clint(p, c, lim=1):
    """clip interval [p-c, p+c] to [-lim, lim] (lim=1 by default)"""
    return np.clip([p - c, p + c], -lim, lim)


def compute_j_grid(resolution):
    """Compute a grid of polarizations with given resolution"""
    size = resolution**2
    thx = np.linspace(0, 2 * np.pi, num=resolution)
    thy = np.linspace(0, 2 * np.pi, num=resolution)

    jones_vecs = np.zeros((2, size), dtype=np.complex128)

    thxmat, thymat = np.meshgrid(thx, thy, sparse=False, indexing="ij")

    jones_vecs[0, :] = np.exp(1j * thxmat.reshape(1, size))
    jones_vecs[1, :] = np.exp(1j * thymat.reshape(1, size))

    return jones_vecs, thxmat, thymat


def compute_k_grid(pointing, resolution, centered, cmin):
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
