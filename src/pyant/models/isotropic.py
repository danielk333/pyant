#!/usr/bin/env python

import numpy as np
from numpy.typing import NDArray

from ..beam import Beam


class Isotropic(Beam):
    def __init__(self):
        super().__init__()

    def gain(self, k: NDArray, polarization: NDArray | None = None):
        k_len = k.shape[1] if len(k.shape) > 1 else 0
        if k_len == 0:
            return np.float64(1.0)
        else:
            return np.ones((k_len,), dtype=np.float64)

    def copy(self):
        return Isotropic()
