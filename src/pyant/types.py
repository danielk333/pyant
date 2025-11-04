"""
This module contains convenient type information so that typing can be precise
but not too verbose in the code itself.
"""

from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass, fields
from typing import TypeVar, Type
import numpy.typing as npt
import numpy as np


class SizeError(ValueError):
    pass


class ParamError(TypeError):
    pass


NDArray_N = npt.NDArray
"(n,) shaped ndarray"

NDArray_M = npt.NDArray
"(n,) shaped ndarray"

NDArray_2 = npt.NDArray
"(2,) shaped ndarray"

NDArray_3 = npt.NDArray
"(3,) shaped ndarray"

NDArray_2xN = npt.NDArray
"(2,n) shaped ndarray"

NDArray_Mx2 = npt.NDArray
"(m,2) shaped ndarray"

NDArray_3xN = npt.NDArray
"(3,n) shaped ndarray"

NDArray_3xNxM = npt.NDArray
"(3,n,m) shaped ndarray"

NDArray_Mx2xN = npt.NDArray
"(m,2,n) shaped ndarray"

NDArray_MxM = npt.NDArray
"(m,m) shaped ndarray"

NDArray_MxN = npt.NDArray
"(m,n) shaped ndarray"

P = TypeVar("P", bound="Parameters")


@dataclass
class Parameters:
    """Definition of all the different parameters for the different models.

    The type of an arbitrary parameter can be considered 'scalar' or 'vector'. If it is
    vector, it is a `NDArray` and has one additional axis. If it is scalar, it has the
    basic dimension given as the second element of the tuple or is a float if this is
    `None`. The second argument contains the base shape of the parameter when it is
    scalar, e.g. 'pointing' is a `(3,)` vector when it is a scalar.
    """

    def copy(self: P) -> P:
        kwargs = {key: deepcopy(getattr(self, key)) for key in self.keys}
        return self.__class__(**kwargs)

    @property
    def keys(self) -> list[str]:
        return [key.name for key in fields(self)]

    @classmethod
    def replace_and_broadcast(
        cls: Type[P],
        parameters: P,
        new_parameters: dict[str, npt.NDArray],
    ) -> P:
        data = {
            key.name: [getattr(parameters, key.name), getattr(parameters, key.name + "_shape")]
            for key in fields(parameters)
        }
        assert all(
            [param in data for param in new_parameters]
        ), f"A requested new parameter ({new_parameters.keys()}) does not exist in given parameters"
        assert len(new_parameters) > 0, "empty new parameters, cannot broadcast"

        # they should all be the same size, so lets just pick the first
        key0 = list(new_parameters.keys())[0]
        vector_len = new_parameters[key0].shape[-1]
        for key, (val, shape) in data.items():
            if key in new_parameters:
                data[key][0] = new_parameters[key]
                assert (
                    new_parameters[key].shape[-1] == vector_len
                ), "all new parameters must be vectorized with the same len"
                continue

            if shape is not None:
                data[key][0] = np.broadcast_to(val.reshape((*shape, 1)), (*shape, vector_len))
            else:
                data[key][0] = np.full((vector_len,), val, dtype=np.float64)
        data = {key: val[0] for key, val in data.items()}
        return cls(**data)

    def to_npz(self, path: Path | str):
        """Write defining parameters to a numpy npz file"""
        data = {key.name: getattr(self, key.name) for key in fields(self)}
        np.savez(path, **data)

    @classmethod
    def from_npz(cls: Type[P], path: Path | str) -> P:
        """Load defining parameters from a numpy npz file and instantiate parameters"""
        with np.load(path) as dat:
            data = {key: dat[key] for key in dat}
        data = {key: x[()] if x.ndim == 0 else x for key, x in data.items()}
        return cls(**data)

    def size(self) -> int | None:
        """Return the size of the additional vectorized axis"""
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
