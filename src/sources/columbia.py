"""
Dataset source loader for Columbia Gray & Uncompressed Image Splicing Detection Benchmark.
DVMM Laboratory, Columbia University.
"""

import os
from typing import Iterator, Optional

from ..downloader import download_dataset
from ..utils import calculate_mask_ratio, create_black_mask_png, get_image_dimensions
from .base import BaseDatasetSource, DataRecord


class ColumbiaSource(BaseDatasetSource):
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = (
            data_dir
            or os.environ.get("COLUMBIA_DIR")
            or os.path.join(os.getcwd(), "data", "columbia")
            or os.path.join(os.getcwd(), "data", "Columbia")
        )

    @property
    def name(self) -> str:
        return "Columbia_Gray"

    def _ensure_data_available(self) -> bool:
        if os.path.exists(self.data_dir) and len(os.listdir(self.data_dir)) > 0:
            return True
        try:
            print(f"[{self.name}] Triggering automated download via standard benchmark bundle...")
            downloaded = download_dataset("columbia", target_dir="data")
            if os.path.exists(downloaded):
                self.data_dir = downloaded
                return True
        except Exception as e:
            print(f"[{self.name}] Automated download failed ({e}).")
        return False

    def stream_records(
        self,
        target_real: int = 1000,
        target_fake: int = 1000,
    ) -> Iterator[DataRecord]:
        if not self._ensure_data_available():
            print(f"[{self.name}] Notice: Data directory '{self.data_dir}' not available. Skipping.")
            return

        print(f"[{self.name}] Scanning directory {self.data_dir}...")
        splc_dir = (
            os.path.join(self.data_dir, "4cam_splc")
            if os.path.exists(os.path.join(self.data_dir, "4cam_splc"))
            else os.path.join(self.data_dir, "spliced")
        )
        auth_dir = (
            os.path.join(self.data_dir, "4cam_auth")
            if os.path.exists(os.path.join(self.data_dir, "4cam_auth"))
            else os.path.join(self.data_dir, "authentic")
        )
        mask_dir = os.path.join(self.data_dir, "mask")

        yielded_fake = 0
        yielded_real = 0

        # 1. Fake (spliced) records
        if os.path.exists(splc_dir):
            for fname in sorted(os.listdir(splc_dir)):
                if yielded_fake >= target_fake:
                    break
                if not fname.lower().endswith((".bmp", ".tif", ".jpg", ".png")):
                    continue

                img_path = os.path.join(splc_dir, fname)
                base_name, _ = os.path.splitext(fname)

                mask_path = None
                if os.path.exists(mask_dir):
                    for ext in [".jpg", ".png", ".bmp", ".tif"]:
                        cand = os.path.join(mask_dir, f"{base_name}{ext}")
                        cand_gt = os.path.join(mask_dir, f"{base_name}_mask{ext}")
                        if os.path.exists(cand):
                            mask_path = cand
                            break
                        if os.path.exists(cand_gt):
                            mask_path = cand_gt
                            break

                try:
                    with open(img_path, "rb") as f:
                        img_bytes = f.read()

                    if mask_path and os.path.exists(mask_path):
                        with open(mask_path, "rb") as f:
                            mask_bytes = f.read()
                    else:
                        w, h = get_image_dimensions(img_bytes)
                        mask_bytes = create_black_mask_png(w, h)

                    ratio = calculate_mask_ratio(mask_bytes)
                    if not (0.0005 <= ratio <= 0.95):
                        continue

                    yield DataRecord(
                        id=f"columbia_{base_name}",
                        image_bytes=img_bytes,
                        image_path=fname,
                        mask_bytes=mask_bytes,
                        mask_path=os.path.basename(mask_path) if mask_path else f"{base_name}_mask.png",
                        label=1,
                        label_text="fake",
                        dataset="Columbia_Gray",
                        method="Splicing",
                        mask_ratio=ratio,
                    )
                    yielded_fake += 1
                except Exception:
                    continue

        # 2. Authentic records
        if os.path.exists(auth_dir):
            for fname in sorted(os.listdir(auth_dir)):
                if yielded_real >= target_real:
                    break
                if not fname.lower().endswith((".bmp", ".tif", ".jpg", ".png")):
                    continue

                img_path = os.path.join(auth_dir, fname)
                base_name, _ = os.path.splitext(fname)
                try:
                    with open(img_path, "rb") as f:
                        img_bytes = f.read()
                    w, h = get_image_dimensions(img_bytes)
                    black_mask = create_black_mask_png(w, h)

                    yield DataRecord(
                        id=f"columbia_real_{base_name}",
                        image_bytes=img_bytes,
                        image_path=fname,
                        mask_bytes=black_mask,
                        mask_path=f"real_{base_name}_mask.png",
                        label=0,
                        label_text="real",
                        dataset="Columbia_Gray",
                        method="authentic",
                        mask_ratio=0.0,
                    )
                    yielded_real += 1
                except Exception:
                    continue

        print(f"[{self.name}] Completed: {yielded_real} real, {yielded_fake} fake.")
