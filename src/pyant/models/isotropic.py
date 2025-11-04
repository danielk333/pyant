#!/usr/bin/env python

from dataclasses import dataclass

import numpy as np

from ..beam import Beam, get_and_validate_k_shape
from ..types import NDArray_3, NDArray_3xN, NDArray_N, Parameters


@dataclass
class IsotropicParams(Parameters):
    """placeholder"""
    pass


class Isotropic(Beam[IsotropicParams]):
    def gain(self, k: NDArray_3xN | NDArray_3, parameters: IsotropicParams) -> NDArray_N | float:
        size = parameters.size()
        k_len = get_and_validate_k_shape(size, k)
        if k_len is not None:
            return np.ones((k_len,), dtype=np.float64)
        else:
            return np.float64(1.0)

    def copy(self):
        return Isotropic()
