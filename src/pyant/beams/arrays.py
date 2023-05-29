import numpy as np

from ..models import Array
from .. import coordinates


def equidistant_archimedian_spiral(
    antenna_num,
    arc_separation,
    range_coefficient,
    frequency,
    azimuth=0,
    elevation=90,
    degrees=True,
):
    # https://math.stackexchange.com/a/2216736
    antennas = np.zeros((3, antenna_num))
    for ind in range(1, antenna_num):
        d_theta = arc_separation / np.sqrt(1 + antennas[0, ind - 1] ** 2)
        antennas[0, ind] = antennas[0, ind - 1] + d_theta
        antennas[2, ind] = range_coefficient * antennas[0, ind]

    antennas = coordinates.sph_to_cart(antennas, degrees=False)
    antennas = antennas.reshape((3, 1, antenna_num))

    return Array(
        azimuth=azimuth,
        elevation=elevation,
        frequency=frequency,
        antennas=antennas,
        degrees=degrees,
    )
