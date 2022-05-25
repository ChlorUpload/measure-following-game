# -*- coding: utf-8 -*-

import numpy as np
from os import PathLike
from pathlib import Path
from typing import Literal


__all__ = ["ContextRenderer"]


class ContextRenderer(object):
    def __init__(self, score_root: str | Path | PathLike):
        self.score_root = Path(score_root)
        self.score_musicxml_path = self.score_root / "score.musicxml"

        if not self.score_musicxml_path.is_file():
            raise FileNotFoundError()

    def reset(self, measure_range: tuple[int, int]):
        ...

    def render(
        self,
        measure_range: tuple[int, int],
        similarity_matrix: np.ndarray,
        true_measure: int | None = None,
        pred_measure: int | None = None,
        mode: Literal["human", "rgb_array"] = "human",
    ) -> np.ndarray | None:
        similarities = [float(similarity) for similarity in similarity_matrix[:, 0]]

    def close(self):
        ...
