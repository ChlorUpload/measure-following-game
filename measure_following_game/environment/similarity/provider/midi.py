# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path

from .base import SimilarityProviderBase


__all__ = ["MIDISimilarityProvider"]


class MIDISimilarityProvider(SimilarityProviderBase):
    def __init__(self, path, window_size: int = 10):
        super(MIDISimilarityProvider, self).__init__(window_size)
        self.path = Path(path)

    def _slide_or_stay(self):
        ...

    def _step_core(self) -> tuple[np.ndarray, int, bool, dict]:
        ...

    def _reset_core(
        self,
        *,
        seed: int | None = None,
        return_info: bool = False,
        options: dict | None = None
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        ...

    def render(self, mode: str):
        ...

    def close(self):
        ...
