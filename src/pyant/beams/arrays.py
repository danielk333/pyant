import numpy as np
import spacecoords.spherical as sph
from ..models import Array


def equidistant_archimedian_spiral(
    antenna_num: int,
    arc_separation: float,
    range_coefficient: float,
    degrees: bool = True,
    array_kwargs: dict | None = None,
) -> Array:
    # https://math.stackexchange.com/a/2216736
    antennas_ang = np.zeros((3, antenna_num))
    for ind in range(1, antenna_num):
        d_theta = arc_separation / np.sqrt(1 + antennas_ang[0, ind - 1] ** 2)
        antennas_ang[0, ind] = antennas_ang[0, ind - 1] + d_theta
        antennas_ang[2, ind] = range_coefficient * antennas_ang[0, ind]

    antennas = sph.sph_to_cart(antennas_ang, degrees=False)
    antennas = antennas.reshape((3, antenna_num, 1))

    if array_kwargs is None:
        array_kwargs = dict()

    return Array(antennas=antennas, **array_kwargs)


def circular_array(
    array_radius: float,
    antenna_spacing: float,
    array_kwargs: dict | None = None,
) -> Array:
    xmat, ymat = np.meshgrid(
        np.arange(-array_radius, array_radius + antenna_spacing, antenna_spacing),
        np.arange(-array_radius, array_radius + antenna_spacing, antenna_spacing),
    )
    xmat = xmat.reshape((xmat.size,))
    ymat = ymat.reshape((ymat.size,))
    filt = xmat**2 + ymat**2 < array_radius**2
    xmat = xmat[filt]
    ymat = ymat[filt]
    antenna_num = len(xmat)
    antennas = np.stack([xmat, ymat, np.zeros_like(ymat)], axis=0)
    antennas = antennas.reshape((3, antenna_num, 1))

    if array_kwargs is None:
        array_kwargs = dict()

    return Array(antennas=antennas, **array_kwargs)
