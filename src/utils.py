"""
Utility functions for AI image tampering dataset processing.
Provides pure-Python helpers for image dimension inspection,
black mask synthesis, and pixel-level tampering ratio calculation.
"""

import io
import struct
import zlib
from typing import Optional, Tuple


def get_image_dimensions(data: bytes) -> Tuple[int, int]:
    """
    Extracts (width, height) from raw image bytes (PNG or JPEG).
    Defaults to (512, 512) if dimensions cannot be determined.
    """
    if not data or len(data) < 24:
        return 512, 512

    # PNG format
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        try:
            w, h = struct.unpack(">II", data[16:24])
            return int(w), int(h)
        except Exception:
            return 512, 512

    # JPEG format
    if data[:2] == b"\xff\xd8":
        idx = 2
        data_len = len(data)
        while idx < data_len - 8:
            if data[idx] != 0xFF:
                idx += 1
                continue
            marker = data[idx + 1]
            # Baseline, Extended sequential, Progressive DCT markers
            if marker in [0xC0, 0xC1, 0xC2, 0xC3]:
                h, w = struct.unpack(">HH", data[idx + 5 : idx + 9])
                return int(w), int(h)
            elif marker in [0xD9, 0xDA]:
                break
            else:
                length = struct.unpack(">H", data[idx + 2 : idx + 4])[0]
                idx += 2 + length

    return 512, 512


def create_black_mask_png(width: int, height: int) -> bytes:
    """
    Generates a valid 8-bit single-channel grayscale PNG filled with 0s (pure black).
    Used as ground truth mask for real (untampered) images.
    """
    width = max(1, int(width))
    height = max(1, int(height))

    # Construct uncompressed raster with scanline filter type 0 (None)
    raw_data = bytearray()
    scanline_len = width
    zero_line = b"\x00" * scanline_len
    for _ in range(height):
        raw_data.append(0)  # filter type 0
        raw_data.extend(zero_line)

    compressed = zlib.compress(bytes(raw_data), level=6)

    def _chunk(chunk_type: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + payload)
        return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)

    header = b"\x89PNG\r\n\x1a\n"
    # IHDR: width, height, bit_depth=8, color_type=0 (grayscale), compression=0, filter=0, interlace=0
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    idat = _chunk(b"IDAT", compressed)
    iend = _chunk(b"IEND", b"")
    return header + ihdr + idat + iend


def calculate_mask_ratio(mask_bytes: bytes, threshold: int = 10) -> float:
    """
    Computes the ratio of non-zero (tampered) pixels in a mask.
    Returns a float between 0.0 (all authentic / black) and 1.0 (all tampered / white).
    Uses Pillow to accurately decode JPEG, PNG, BMP masks and avoid compression noise.
    """
    if not mask_bytes or len(mask_bytes) < 8:
        return 0.0

    # 1. Primary: Fast and accurate PIL decoding
    try:
        from PIL import Image

        with Image.open(io.BytesIO(mask_bytes)) as img:
            gray = img.convert("L")
            hist = gray.histogram()
            total_pixels = sum(hist)
            if total_pixels == 0:
                return 0.0
            # Pixels > threshold are considered tampered (avoids JPEG 8x8 DCT compression noise near 0)
            tampered = sum(hist[threshold + 1 :])
            return float(tampered) / float(total_pixels)
    except Exception:
        pass

    # 2. Fallback: Pure-Python PNG format parser
    if mask_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        try:
            w, h, depth, ctype = struct.unpack(">IIBB", mask_bytes[16:26])
            idat_parts = []
            idx = 8
            total_len = len(mask_bytes)
            while idx < total_len - 12:
                length = struct.unpack(">I", mask_bytes[idx : idx + 4])[0]
                chunk_type = mask_bytes[idx + 4 : idx + 8]
                if chunk_type == b"IDAT":
                    idat_parts.append(mask_bytes[idx + 8 : idx + 8 + length])
                elif chunk_type == b"IEND":
                    break
                idx += 12 + length

            if not idat_parts:
                return 0.0

            raw = zlib.decompress(b"".join(idat_parts))
            bpp = 1 if ctype == 0 else (3 if ctype == 2 else (4 if ctype == 6 else 1))
            stride = 1 + w * bpp
            non_zero = 0
            total_pixels = w * h
            for y in range(h):
                offset = y * stride + 1
                line = raw[offset : offset + w * bpp]
                if bpp == 1:
                    non_zero += sum(1 for p in line if p > threshold)
                else:
                    for i in range(0, len(line), bpp):
                        if any(p > threshold for p in line[i : i + bpp]):
                            non_zero += 1

            return float(non_zero) / float(total_pixels) if total_pixels > 0 else 0.0
        except Exception:
            pass

    # 3. Final fallback estimation based on raw byte values
    non_zero = sum(1 for b in mask_bytes if b > threshold)
    return min(1.0, non_zero / max(1, len(mask_bytes)))


def is_valid_localized_fake(mask_bytes: Optional[bytes], min_ratio: float = 0.0005, max_ratio: float = 0.95) -> bool:
    """
    Validates whether an image represents a localized tampering / inpainting.
    Avoids 100% fake (full generation) images as well as completely untouched images.
    """
    if mask_bytes is None:
        return False
    ratio = calculate_mask_ratio(mask_bytes)
    return min_ratio <= ratio <= max_ratio
