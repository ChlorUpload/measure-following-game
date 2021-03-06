# -*- coding: utf-8 -*-

__all__ = ["MIDIContextManager"]

from typing import ClassVar

from beartype import beartype
from sabanamusic.common.types import PositiveInt
from sabanamusic.models.musical import MIDIRecord
from sabanamusic.similarity.algorithms import *
from sabanamusic.similarity.utils import get_actual_alignment

from measure_following_game.environment.context.manager.base import ContextManager
from measure_following_game.environment.context.renderer import ContextRenderer


class MIDIContextManager(ContextManager):

    num_features: ClassVar[PositiveInt] = 3

    @beartype
    def __init__(
        self,
        renderer: ContextRenderer,
        record: MIDIRecord,
        window_size: PositiveInt = 16,
        memory_size: PositiveInt = 16,
        **kwargs,
    ):
        super(MIDIContextManager, self).__init__(
            renderer, record, window_size, memory_size
        )

    def _fill_similarity_matrix(self):
        record: MIDIRecord = self.record
        onset_only = record.onset_only
        record_num_frames = record.num_frames
        record_repr_sequence = record.get_repr_sequence()
        record_onset_indices = record.onset_indices

        similarity_matrix = self.similarity_matrix
        similarity_matrix.fill(0.0)
        for idx, measure in enumerate(self.window_measures):
            timewarping_distance, (head, tail) = dtw_compact(
                measure.repr_sequence, record_repr_sequence, subsequence=True
            )

            if onset_only:
                (head, tail) = get_actual_alignment((head, tail), record_onset_indices)

            record_pitch_histogram = record.get_pitch_histogram((head, tail))
            euclidean_distance = euclidean(
                measure.pitch_histogram, record_pitch_histogram
            )

            # similarity, subsequence offset, sebsequence size
            similarity_matrix[idx, 0] = calc_algorithmic_similarity(
                distances=[timewarping_distance, euclidean_distance], scales=[1.0, 1.0]
            )
            similarity_matrix[idx, 1] = head / (record_num_frames - 1)
            similarity_matrix[idx, 2] = (tail - head) / record_num_frames
