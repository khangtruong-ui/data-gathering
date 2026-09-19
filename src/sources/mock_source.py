"""
Synthetic mock source generator for fast, zero-bandwidth unit testing and dry runs.
Produces valid PNG images and pixel-accurate localized masks in memory.
"""

import struct
import zlib
from typing import Iterator

from ..utils import create_black_mask_png
from .base import BaseDatasetSource, DataRecord


def _create_mock_png(width: int, height: int, r_seed: int = 42) -> bytes:
    """Generates an RGB PNG image with a smooth gradient pattern."""
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0
        for x in range(width):
            r = (x * 255 // width + r_seed * 17) % 256
            g = (y * 255 // height + r_seed * 31) % 256
            b = ((x + y) * 255 // (width + height) + r_seed * 53) % 256
            raw.extend(bytes([r, g, b]))

    def _chunk(chunk_type: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + payload)
        return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    idat = _chunk(b"IDAT", zlib.compress(bytes(raw)))
    iend = _chunk(b"IEND", b"")
    return header + ihdr + idat + iend


def _create_mock_tampered_pair(
    width: int, height: int, sample_idx: int, method_name: str
) -> tuple[bytes, bytes, float]:
    """
    Generates a tampered RGB PNG and its corresponding ground truth 1-channel mask.
    The tampered patch is placed in a localized quadrant.
    """
    # 1. Authentic background
    raw_img = bytearray()
    raw_mask = bytearray()

    # Localized patch coordinates (e.g. 20% to 50% width and height)
    x1, x2 = int(width * 0.25), int(width * 0.55)
    y1, y2 = int(height * 0.25), int(height * 0.55)
    patch_pixels = (x2 - x1) * (y2 - y1)
    total_pixels = width * height
    ratio = patch_pixels / total_pixels

    for y in range(height):
        raw_img.append(0)
        raw_mask.append(0)
        for x in range(width):
            if x1 <= x < x2 and y1 <= y < y2:
                # Tampered region
                raw_mask.append(255)
                # Inpainted / tampered color
                raw_img.extend(bytes([220, 50, (sample_idx * 40) % 256]))
            else:
                # Authentic background
                raw_mask.append(0)
                r = (x * 255 // width) % 256
                g = (y * 255 // height) % 256
                b = ((x + y) * 255 // (width + height)) % 256
                raw_img.extend(bytes([r, g, b]))

    def _chunk(chunk_type: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + payload)
        return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)

    header = b"\x89PNG\r\n\x1a\n"

    # Image PNG (RGB, color type 2)
    img_ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    img_idat = _chunk(b"IDAT", zlib.compress(bytes(raw_img)))
    img_bytes = header + img_ihdr + img_idat + _chunk(b"IEND", b"")

    # Mask PNG (Grayscale, color type 0)
    mask_ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    mask_idat = _chunk(b"IDAT", zlib.compress(bytes(raw_mask)))
    mask_bytes = header + mask_ihdr + mask_idat + _chunk(b"IEND", b"")

    return img_bytes, mask_bytes, ratio


class MockDatasetSource(BaseDatasetSource):
    """Generates synthetic triplets for mock tests and end-to-end verification."""

    def __init__(self, dataset_name: str = "Mock_OpenSDID", method: str = "Flux.1"):
        self._name = dataset_name
        self._method = method

    @property
    def name(self) -> str:
        return self._name

    def stream_records(
        self,
        target_real: int = 1000,
        target_fake: int = 1000,
    ) -> Iterator[DataRecord]:
        width, height = 256, 256

        # Generate fake records
        for i in range(target_fake):
            img_b, mask_b, ratio = _create_mock_tampered_pair(
                width, height, sample_idx=i, method_name=self._method
            )
            yield DataRecord(
                id=f"{self._name.lower()}_fake_{i:04d}",
                image_bytes=img_b,
                image_path=f"{self._name.lower()}_fake_{i:04d}.png",
                mask_bytes=mask_b,
                mask_path=f"{self._name.lower()}_fake_{i:04d}_mask.png",
                label=1,
                label_text="fake",
                dataset=self._name,
                method=self._method,
                mask_ratio=ratio,
            )

        # Generate real records
        for i in range(target_real):
            real_img = _create_mock_png(width, height, r_seed=i + 100)
            black_mask = create_black_mask_png(width, height)
            yield DataRecord(
                id=f"{self._name.lower()}_real_{i:04d}",
                image_bytes=real_img,
                image_path=f"{self._name.lower()}_real_{i:04d}.png",
                mask_bytes=black_mask,
                mask_path=f"{self._name.lower()}_real_{i:04d}_mask.png",
                label=0,
                label_text="real",
                dataset=self._name,
                method="authentic",
                mask_ratio=0.0,
            )
