"""
Dataset stream reader for CocoGlide (TruFor CVPR 2023 benchmark).
GLIDE-inpainted localized tampering on MS-COCO backgrounds.
"""

import io
import os
import urllib.request
import zipfile
from typing import Iterator

from ..utils import calculate_mask_ratio, create_black_mask_png, get_image_dimensions
from .base import BaseDatasetSource, DataRecord


class RemoteZipFile(io.RawIOBase):
    """Seekable HTTP Range request wrapper for zip files to stream without full download."""

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            self.length = int(resp.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.length + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, b: bytearray) -> int:
        size = len(b)
        if self.pos >= self.length:
            return 0
        end = min(self.pos + size - 1, self.length - 1)
        req = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={self.pos}-{end}", "User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req) as resp:
            chunk = resp.read()
            b[: len(chunk)] = chunk
            self.pos += len(chunk)
            return len(chunk)


class CocoGlideSource(BaseDatasetSource):
    ZIP_URL = "https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip"

    @property
    def name(self) -> str:
        return "CocoGlide"

    def stream_records(
        self,
        target_real: int = 1000,
        target_fake: int = 1000,
    ) -> Iterator[DataRecord]:
        print(f"[{self.name}] Connecting to remote archive {self.ZIP_URL}...")
        rz = RemoteZipFile(self.ZIP_URL)
        zf = zipfile.ZipFile(rz)

        namelist = zf.namelist()
        real_files = sorted([n for n in namelist if n.startswith("real/") and not n.endswith("/")])
        fake_files = sorted([n for n in namelist if n.startswith("fake/") and not n.endswith("/")])
        mask_files = sorted([n for n in namelist if n.startswith("mask/") and not n.endswith("/")])

        # Map COCO IDs for mask lookup
        # e.g., mask/airplane_139871_mask.png -> 139871
        mask_map = {}
        for m in mask_files:
            base = os.path.basename(m).replace("_mask.png", "")
            parts = base.split("_")
            if len(parts) >= 2:
                coco_id = parts[-1]
                mask_map[coco_id] = m

        yielded_real = 0
        yielded_fake = 0

        # Stream fake records first
        for f in fake_files:
            if yielded_fake >= target_fake:
                break
            # e.g. fake/glide_inpainting_val2017_139871_up.png
            base = os.path.basename(f)
            coco_id = base.split("_")[3] if len(base.split("_")) > 3 else None
            if not coco_id or coco_id not in mask_map:
                continue

            mask_path = mask_map[coco_id]
            fake_bytes = zf.read(f)
            mask_bytes = zf.read(mask_path)

            ratio = calculate_mask_ratio(mask_bytes)
            # Avoid 100% fake (full synthesis) and untouched images
            if not (0.0005 <= ratio <= 0.95):
                continue

            yield DataRecord(
                id=f"cocoglide_{coco_id}",
                image_bytes=fake_bytes,
                image_path=os.path.basename(f),
                mask_bytes=mask_bytes,
                mask_path=os.path.basename(mask_path),
                label=1,
                label_text="fake",
                dataset="CocoGlide",
                method="GLIDE",
                mask_ratio=ratio,
            )
            yielded_fake += 1

        # Stream real records
        for r in real_files:
            if yielded_real >= target_real:
                break
            real_bytes = zf.read(r)
            base = os.path.basename(r)
            w, h = get_image_dimensions(real_bytes)
            black_mask = create_black_mask_png(w, h)

            coco_id = base.split("_")[-1].replace(".png", "")
            yield DataRecord(
                id=f"cocoglide_real_{coco_id}",
                image_bytes=real_bytes,
                image_path=base,
                mask_bytes=black_mask,
                mask_path=f"real_{coco_id}_mask.png",
                label=0,
                label_text="real",
                dataset="CocoGlide",
                method="authentic",
                mask_ratio=0.0,
            )
            yielded_real += 1

        print(f"[{self.name}] Completed streaming: {yielded_real} real, {yielded_fake} fake.")
