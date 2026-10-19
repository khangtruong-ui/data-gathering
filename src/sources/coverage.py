"""
Dataset source loader for COVERAGE (Copy-Move Forgery Database with Similar but Genuine Objects).
Wen et al., ICIP 2016.
"""

import os
from typing import Iterator, Optional

from ..downloader import download_dataset
from ..utils import calculate_mask_ratio, create_black_mask_png, get_image_dimensions
from .base import BaseDatasetSource, DataRecord


class CoverageSource(BaseDatasetSource):
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = (
            data_dir
            or os.environ.get("COVERAGE_DIR")
            or os.path.join(os.getcwd(), "data", "Coverage")
            or os.path.join(os.getcwd(), "data", "coverage")
        )

    @property
    def name(self) -> str:
        return "Coverage"

    def _ensure_data_available(self) -> bool:
        if os.path.exists(self.data_dir) and len(os.listdir(self.data_dir)) > 0:
            return True
        try:
            print(f"[{self.name}] Triggering automated download via standard benchmark bundle...")
            downloaded = download_dataset("coverage", target_dir="data")
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
        img_dir = (
            os.path.join(self.data_dir, "image")
            if os.path.exists(os.path.join(self.data_dir, "image"))
            else self.data_dir
        )
        mask_dir = os.path.join(self.data_dir, "mask")

        yielded_fake = 0
        yielded_real = 0

        if not os.path.exists(img_dir):
            return

        all_files = sorted(os.listdir(img_dir))

        for fname in all_files:
            if yielded_fake >= target_fake and yielded_real >= target_real:
                break
            if not fname.lower().endswith((".tif", ".png", ".jpg")):
                continue

            img_path = os.path.join(img_dir, fname)
            base_name, _ = os.path.splitext(fname)

            # In COVERAGE:
            # e.g., '1t.tif' is forged copy-move image, '1.tif' is authentic original
            # mask is '1forged.tif' in mask/
            is_forged = base_name.endswith("t") or "forged" in fname.lower()
            mask_base = base_name[:-1] if base_name.endswith("t") else base_name

            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()

                # Process fake
                if is_forged and yielded_fake < target_fake:
                    mask_path = None
                    if os.path.exists(mask_dir):
                        candidates = [
                            os.path.join(mask_dir, f"{mask_base}forged.tif"),
                            os.path.join(mask_dir, f"{mask_base}forged.png"),
                            os.path.join(mask_dir, fname),
                            os.path.join(mask_dir, f"{base_name}_mask.png"),
                            os.path.join(mask_dir, f"{base_name}.png"),
                            os.path.join(mask_dir, f"{base_name}.tif"),
                        ]
                        for c in candidates:
                            if os.path.exists(c):
                                mask_path = c
                                break

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
                        id=f"coverage_{base_name}",
                        image_bytes=img_bytes,
                        image_path=fname,
                        mask_bytes=mask_bytes,
                        mask_path=os.path.basename(mask_path) if mask_path else f"{base_name}_mask.png",
                        label=1,
                        label_text="fake",
                        dataset="Coverage",
                        method="CopyMove",
                        mask_ratio=ratio,
                    )
                    yielded_fake += 1

                # Process authentic
                elif not is_forged and yielded_real < target_real:
                    w, h = get_image_dimensions(img_bytes)
                    black_mask = create_black_mask_png(w, h)

                    yield DataRecord(
                        id=f"coverage_real_{base_name}",
                        image_bytes=img_bytes,
                        image_path=fname,
                        mask_bytes=black_mask,
                        mask_path=f"real_{base_name}_mask.png",
                        label=0,
                        label_text="real",
                        dataset="Coverage",
                        method="authentic",
                        mask_ratio=0.0,
                    )
                    yielded_real += 1
            except Exception:
                continue

        print(f"[{self.name}] Completed: {yielded_real} real, {yielded_fake} fake.")
