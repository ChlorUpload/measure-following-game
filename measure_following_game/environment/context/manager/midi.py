# -*- coding: utf-8 -*-

__all__ = ["MIDIContextManager"]

from typing import ClassVar

from beartype import beartype
from sabanamusic.common import MIDIRecord
from sabanamusic.similarity import *
from sabanamusic.typing import PositiveInt

from measure_following_game.environment.context.manager.base import ContextManager


class MIDIContextManager(ContextManager):

    num_features: ClassVar[PositiveInt] = 3

    @beartype
    def __init__(
        self,
        renderer: ContextManager,
        record: MIDIRecord,
        window_size: PositiveInt = 32,
        memory_size: PositiveInt = 32,
    ):
        super(MIDIContextManager, self).__init__(
            renderer, record, window_size, memory_size
        )

    def _fill_similarity_matrix(self):
        onset_only = self.renderer.onset_only

        record: MIDIRecord = self.record
        record_num_frames = record.num_frames
        record_repr_sequence, record_onset_indices = record.get_repr_sequence()

        similarity_matrix = self.similarity_matrix
        similarity_matrix.fill(0.0)
        for idx, measure in enumerate(self.window_measures):
            timewarping_distance, (head, tail) = dtw_compact(
                measure.repr_sequence, record_repr_sequence, subsequence=True
            )

            if onset_only:
                (head, tail) = get_alignment_without_padding((head, tail))

            record_pitch_histogram = record.get_pitch_histogram((head, tail))
            euclidean_distance = euclidean(
                measure.pitch_histogram, record_pitch_histogram
            )

            # similarity, offset, size
            similarity_matrix[idx, 0] = calc_algorithmic_similarity(
                distances=[timewarping_distance, euclidean_distance]
            )
            similarity_matrix[idx, 1] = head / (record_num_frames - 1)
            similarity_matrix[idx, 2] = (tail - head) / record_num_frames
