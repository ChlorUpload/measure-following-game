# -*- coding: utf-8 -*-

__all__ = ["ActType", "ObsType"]

from typing import Annotated

from beartype.vale import *
import numpy as np
import numpy.typing as npt


def _is_normalized(ndarray: np.ndarray) -> bool:
    return np.all(np.logical_and(0.0 <= ndarray, ndarray <= 1.0))


def _is_normalized_matrix(ndarray: np.ndarray) -> bool:
    return ndarray.ndim == 2 and _is_normalized(ndarray)


def _is_policy(ndarray: np.ndarray) -> bool:
    return (
        ndarray.ndim == 1
        and _is_normalized(ndarray)
        and np.isclose(np.sum(ndarray), 1.0)
    )


def _is_policy_matrix(ndarray: np.ndarray) -> bool:
    assert ndarray.ndim == 2
    for row in ndarray:
        if not _is_policy(row):
            return False
    return True


ActType = Annotated[npt.NDArray[np.float32], Is[_is_policy]]
ObsType = tuple[
    Annotated[npt.NDArray[np.float32], Is[_is_normalized_matrix]],
    Annotated[npt.NDArray[np.float32], Is[_is_policy_matrix]],
]
