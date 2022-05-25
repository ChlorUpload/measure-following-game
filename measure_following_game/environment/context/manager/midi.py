# -*- coding: utf-8 -*-

from beartype import beartype
from sabanamusic.common import MIDIRecord
from sabanamusic.similarity import *
from sabanamusic.typing import PathLike, PositiveFloat, PositiveInt
from typing import ClassVar

from measure_following_game.environment.context.manager.base import ContextManagerBase
from measure_following_game.environment.context.renderer import ContextRenderer


__all__ = ["MIDIContextManager"]


class MIDIContextManager(ContextManagerBase):

    num_features: ClassVar[PositiveInt] = 3

    @beartype
    def __init__(
        self,
        score_root: PathLike,
        record: MIDIRecord,
        renderer: ContextRenderer,
        *,
        fps: PositiveInt = 20,
        onset_only: bool = True,
        window_size: PositiveInt = 32,
        threshold: PositiveFloat = 0.33,
    ):
        super(MIDIContextManager, self).__init__(
            score_root,
            record,
            renderer,
            fps=fps,
            onset_only=onset_only,
            window_size=window_size,
        )
        self.threshold = threshold

    def _fill_similarity_matrix(self):
        similarity_matrix = self.similarity_matrix
        similarity_matrix.fill(0.0)

        record: MIDIRecord = self.record
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
            euclidean_distance = euclidean(
                measure.pitch_histogram, record_pitch_histogram
            )

            # similarity, offset, size
            similarity_matrix[idx, 0] = calc_algorithmic_similarity(
                (timewarping_distance, euclidean_distance), threshold
            )
            similarity_matrix[idx, 1] = head / (record_num_frames - 1)
            similarity_matrix[idx, 2] = (tail - head) / record_num_frames
