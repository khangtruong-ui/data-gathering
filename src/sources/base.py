"""
Base classes and interfaces for dataset streaming sources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class DataRecord:
    id: str
    image_bytes: bytes
    image_path: str
    mask_bytes: bytes
    mask_path: str
    label: int  # 0: real, 1: fake
    label_text: str  # "real" or "fake"
    dataset: str
    method: str
    mask_ratio: float


class BaseDatasetSource(ABC):
    """
    Abstract base class for streaming dataset sources.
    Yields DataRecord items one-by-one without requiring full dataset downloads.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the dataset source."""
        pass

    @abstractmethod
    def stream_records(
        self,
        target_real: int = 1000,
        target_fake: int = 1000,
    ) -> Iterator[DataRecord]:
        """
        Streams up to target_real authentic records and target_fake tampered records.
        """
        pass
