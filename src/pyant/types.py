"""
This module contains convenient type information so that typing can be precise
but not too verbose in the code itself.
"""

from pathlib import Path
from dataclasses import dataclass, fields
from typing import TypeVar, Type
import numpy.typing as npt
import numpy as np


class SizeError(ValueError):
    pass


NDArray_N = npt.NDArray
"(n,) shaped ndarray"

NDArray_2 = npt.NDArray
"(2,) shaped ndarray"

NDArray_3 = npt.NDArray
"(3,) shaped ndarray"

NDArray_2xN = npt.NDArray
"(3,n) shaped ndarray"

NDArray_3xN = npt.NDArray
"(3,n) shaped ndarray"

NDArray_3xNxM = npt.NDArray
"(3,n,m) shaped ndarray"

NDArray_MxM = npt.NDArray
"(m,m) shaped ndarray"

P = TypeVar("P", bound="Parameters")


@dataclass
class Parameters:
    def to_npz(self, path: Path | str):
        """Write defining parameters to a numpy npz file"""
        data = {key.name: getattr(self, key.name) for key in fields(self)}
        np.savez(path, **data)

    @classmethod
    def from_npz(cls: Type[P], path: Path | str) -> P:
        """Load defining parameters from a numpy npz file and instantiate parameters"""
        with np.load(path) as dat:
            data = {key: dat[key] for key in dat}
        return cls(**data)

    def size(self) -> int | None:
        sizes = []
        size: int | None
        for key in fields(self):
            val = getattr(self, key.name)
            val_shape = getattr(self, key.name + "_shape")
            if isinstance(val, np.ndarray):
                if val_shape is None:
                    size = val.size
                else:
                    size = None if len(val.shape) == len(val_shape) else val.shape[-1]
            else:
                size = None
            sizes.append(size)
        if len(sizes) > 0:
            return sizes[0]
        else:
            return None

    def __post_init__(self):
        """Check that all the input parameters line up with the parameter restrictions."""
        sizes = []
        for key in fields(self):
            val = getattr(self, key.name)
            val_shape = getattr(self, key.name + "_shape")
            if isinstance(val, np.ndarray):
                if val_shape is None:
                    if not len(val.shape) == 1:
                        raise SizeError(
                            f"value with {val_shape} shape cannot have {val.shape} shape"
                        )

                    size = val.size
                else:
                    if len(val.shape) != len(val_shape) and len(val.shape) != (len(val_shape) + 1):
                        raise SizeError(
                            f"value with {val_shape} shape cannot have {val.shape} shape"
                        )
                    if val.shape[: len(val_shape)] != val_shape:
                        raise SizeError(
                            f"value with {val_shape} shape cannot have {val.shape} shape"
                        )

                    size = None if len(val.shape) == len(val_shape) else val.shape[-1]
            else:
                size = None
            sizes.append(size)
        if not all(x == sizes[0] for x in sizes):
            raise SizeError(f"all parameter shapes must line up: {sizes}")
