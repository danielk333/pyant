"""Definition of all the different parameters and options for the different models.

The type of an arbitrary parameter can be considered 'scalar' or 'vector'. If it is
vector, it is a `NDArray` and has one additional axis. If it is scalar, it has the
basic dimension given as the second element of the tuple or is a float if this is
`None`. The second argument contains the base shape of the parameter when it is
scalar, e.g. 'pointing' is a `(3,)` vector when it is a scalar.
"""

from dataclasses import dataclass
from typing import ClassVar
from numpy.typing import NDArray
from ..types import NDArray_3xN, NDArray_3, SizeError, Parameters


def get_and_validate_k_shape(param_size: int | None, k: NDArray_3xN | NDArray_3) -> int | None:
    """Helper function to validate the input direction vector shape is correct.
    It returns the size of the second dimension of the k vector, if it does not
    have this dimension then it returns None.
    """
    k_len = k.shape[1] if len(k.shape) > 1 else None

    if len(k.shape) > 2 or k.shape[0] != 3:
        raise SizeError(f"k vector shape cannot be {k.shape}")

    if param_size is not None and k_len is not None and param_size != k_len:
        raise SizeError(f"parameter size {param_size} != k vector size {k.shape}")
    return k_len


@dataclass
class AiryParams(Parameters):
    """
    Parameters
    ----------
    pointing
        Pointing direction of the boresight
    frequency
        Frequency of the radar
    radius
        Radius in meters of the airy disk
    """
    pointing: NDArray | float
    frequency: NDArray | float
    radius: NDArray | float

    pointing_shape: ClassVar[tuple[int, ...]] = (3,)
    frequency_shape: ClassVar[None] = None
    radius_shape: ClassVar[None] = None
