# -*- coding: utf-8 -*-

import numpy as np
from os import PathLike
from pathlib import Path
from sabanamusic.common import MIDIRecordBase
from sabanamusic.similarity import *
from typing import ClassVar, Literal

from measure_following_game.environment.context.manager.base import ContextManagerBase


__all__ = ["MIDIContextManager"]


class MIDIContextManager(ContextManagerBase):

    num_features: ClassVar[int] = 3

    def __init__(
        self,
        score_root: str | Path | PathLike,
        record: MIDIRecordBase,
        *,
        fps: int = 20,
        onset_only: bool = True,
        window_size: int = 10,
        threshold: float = 0.33,
    ):
        super(MIDIContextManager, self).__init__(
            score_root, record, fps=fps, onset_only=onset_only, window_size=window_size
        )
        self.record = record
        self.threshold = threshold  # for normalizing similarities

    def _fill_similarity_matrix(self):
        similarity_matrix = self.similarity_matrix
        similarity_matrix.fill(0.0)

        record = self.record
        record_num_frames = record.num_frames
        record_repr_sequence = record.get_repr_sequence()
        threshold = self.threshold

        for idx, measure in enumerate(self.window):
            timewarping_distance, (head, tail) = dtw_compact(
                measure.repr_sequence, record_repr_sequence, subsequence=True
            )

            if self.onset_only:
               alignment = get_alignment_without_padding(alignment)

            record_pitch_histogram = record.get_pitch_histogram(alignment)
            euclidean_distance = euclidean(measure.pitch_histogram, record_pitch_histogram)

            # similarity, offset, size
            similarity_matrix[idx, 0] = calc_algorithmic_similarity((timewarping_distance, euclidean_distance), threshold)
            similarity_matrix[idx, 1] = head / (record_num_frames - 1)
            similarity_matrix[idx, 2] = (tail - head) / record_num_frames