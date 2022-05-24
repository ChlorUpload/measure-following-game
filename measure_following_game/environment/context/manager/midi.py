# -*- coding: utf-8 -*-

import numpy as np
from os import PathLike
from pathlib import Path
from sabanamusic.common import Measure
from sabanamusic.similarity import *
from typing import ClassVar

from measure_following_game.environment.context.manager.base import ContextManagerBase


__all__ = ["MIDIContextManager"]


class MIDIContextManager(ContextManagerBase):

    metadata: ClassVar[dict] = {"render_modes": ["human"]}
    num_features: ClassVar[int] = 3

    def __init__(
        self,
        score_root: str | Path | PathLike,
        record_name: str,
        *,
        fps: int = 20,
        onset_only: bool = True,
        window_size: int = 10,
        threshold: float = 0.33,
    ):
        super(MIDIContextManager, self).__init__(
            score_root, fps=fps, onset_only=onset_only, window_size=window_size
        )

        self.record_root = self.score_root / record_name
        self.record_midi_path = self.record_root.with_suffix(".midi")
        self.record_annotation_path = self.record_root.with_suffix(".csv")
        self.record = Measure()

        if not self.record_midi_path.is_file():
            raise FileNotFoundError()
        elif not self.record_annotation_path.is_file():
            raise FileNotFoundError()

        self.threshold = threshold

    def _slide_or_stay(self):
        ...

    def _make_similarity_matrix(self) -> np.ndarray:
        similarity_matrix = np.zeros(self.window_shape, dtype=np.float32)

        record = self.record
        # record_num_frames = record.num_frames
        record_num_frames = 60
        record_repr_sequence = record.repr_sequence

        for idx, measure in enumerate(self.window):
            time_warping_distance, (head, tail) = dtw_compact(
                measure.repr_sequence, record_repr_sequence, subsequence=True
            )

            # if self.onset_only:
            #    alignment = get_alignment_without_padding(alignment)

            # record_pitch_histogram = record.get_pitch_histogram(alignment, self.onset_only)
            # euclidean_distance = euclidean(measure.pitch_histogram, record_pitch_histogram)

            euclidean_distance = 0.0
            similarity = time_warping_distance + euclidean_distance
            if similarity != 0:
                similarity = self.threshold / similarity

            offset = head / record_num_frames
            size = (tail - head) / record_num_frames

            similarity_matrix[idx, 0] = similarity
            similarity_matrix[idx, 1] = offset
            similarity_matrix[idx, 2] = size

        return similarity_matrix

    def _step_core(self) -> tuple[np.ndarray, int, dict]:
        if self.done:
            return -np.ones(self.window_shape, dtype=np.float32), -1, {"done": True}
        else:
            # true_measure = self.record.true_measure
            true_measure = 0
            # self.record.step()
            similarity_matrix = self._make_similarity_matrix()
            return similarity_matrix, true_measure, {}

    def _mock_history(self):
        # TODO(kaparoo): FIX HERE
        self.local_history = [0]
        self.global_history = [0]
        self.window_head = 0
        self.num_measures_in_window = self.window_size

    def _reset_core(self, *, **kwargs) -> np.ndarray | tuple[np.ndarray, dict]:
        self._mock_history()
        # record.reset()
        similarity_matrix = self._make_similarity_matrix()
        return similarity_matrix

    def render(self, mode: str):
        match mode:
            case "human":
                ...
            case _:
                raise KeyError(f"Unsupported mode: {mode}")

    def close(self):
        super(MIDIContextManager, self).close()
        del self.record
