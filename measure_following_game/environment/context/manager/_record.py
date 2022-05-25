# -*- coding: utf-8 -*-

from abc import abstractmethod, ABC
from beartype import beartype
import numpy as np
import pandas as pd
from pathlib import Path
from sabanamusic.conversion import *
from sabanamusic.typing import (
    Index,
    MIDIKeySequence,
    MIDIMatrix,
    PathLike,
    PitchHistogram,
)


__all__ = ["RecordBase", "MIDIRecord", "StaticMIDIRecord", "DynamicMIDIRecord"]


class RecordBase(ABC):
    @abstractmethod
    @property
    def done(self) -> bool:
        ...

    @abstractmethod
    @property
    def true_measure(self) -> Index:
        ...

    @abstractmethod
    def step(self, *args, **kwargs):
        ...

    @abstractmethod
    def reset(self, measure_range: tuple[Index, Index], *args, **kwargs):
        ...


class MIDIRecord(RecordBase):
    @beartype
    def __init__(
        self,
        record_root: PathLike,
        *,
        fps: int = 20,
        duration: int = 3,
        step_freq: int = 10,
        onset_only: bool = True,
    ):
        self.record_root = Path(record_root)
        record_midi_path = self.record_root.with_suffix(".midi")
        record_annotation_path = self.record_root.with_suffix(".csv")

        if not record_midi_path.is_file():
            raise FileNotFoundError()
        elif not record_annotation_path.is_file():
            raise FileNotFoundError()

        self.fps = fps
        self.duration = duration
        self.num_frames = self.fps * self.duration
        self.step_freq = step_freq
        self.onset_only = onset_only
        self.repr_sequence: MIDIKeySequence | None = None
        self.pitch_histogram: PitchHistogram | None = None

        record_midi_matrix, _ = convert_midi_to_midi_matrices(
            midi_path=record_midi_path, fps=fps, mark_onset=True, target_channels=0
        )
        self.entire_record: MIDIMatrix = record_midi_matrix
        self.num_entire_frames = record_midi_matrix.shape[1]

        self._cursor: Index = 0
        self.max_cursor: Index = self.num_entire_frames - self.num_frames

        self.record_annotations = pd.read_csv(record_annotation_path)

    @beartype
    @property
    def cursor(self) -> Index:
        return self._cursor

    @beartype
    @cursor.setter
    def cursor(self, val: Index):
        self._cursor = np.clip(val, 0, self.max_cursor)

    @beartype
    @property
    def current_record(self) -> MIDIMatrix:
        return self.entire_record[:, self.cursor : self.cursor + self.num_frames]

    @beartype
    def get_repr_sequence(self) -> MIDIKeySequence:
        self.repr_sequence = convert_midi_matrix_to_repr_sequence(
            self.current_record, onset_only=self.onset_only, padding=True
        )
        return self.repr_sequence

    @beartype
    def get_pitch_histogram(
        self, alignment: tuple[Index, Index] | None = None
    ) -> PitchHistogram:
        if alignment is None:
            sub_record = self.current_record
        else:
            head, tail = alignment
            head = np.clip(head, 0, self.num_frames - 1)
            tail = np.clip(tail, 1, self.num_frames)
            if head > tail:
                head, tail = tail, head
            sub_record = self.current_record[head:tail]

        self.pitch_histogram = convert_midi_matrix_to_pitch_histogram(
            sub_record, self.onset_only
        )
        return self.pitch_histogram


# TODO(kaparoo): need implementation
class StaticMIDIRecord(MIDIRecord):
    @beartype
    @property
    def done(self) -> bool:
        return self.cursor > self.max_cursor

    @beartype
    @property
    def true_measure(self) -> Index:
        ...

    @beartype
    def step(self):
        self.cursor += self.step_freq

    @beartype
    def reset(self, measure_range: tuple[Index, Index], *args, **kwargs):
        head, tail = measure_range


# TODO(kaparoo): need implementation
class DynamicMIDIRecord(MIDIRecord):
    @beartype
    def __init__(
        self,
        record_root: PathLike,
        *,
        fps: int = 20,
        duration: int = 3,
        step_freq: int = 10,
        onset_only: bool = True,
    ):
        super(DynamicMIDIRecord, self).__init__(
            record_root,
            fps=fps,
            duration=duration,
            step_freq=step_freq,
            onset_only=onset_only,
        )
        self._done = False

    @beartype
    @property
    def done(self) -> bool:
        return self._done

    @beartype
    @property
    def true_measure(self) -> Index:
        ...

    @beartype
    def step(self, measure_range: tuple[Index, Index], *args, **kwargs):
        ...

    @beartype
    def reset(self, measure_range: tuple[Index, Index], *args, **kwargs):
        ...
