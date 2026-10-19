"""
Dataset source loader for AutoSplice (CVPRW 2023 DALL-E 2 prompt-manipulated dataset).
Jia et al., University at Buffalo.
"""

import os
from typing import Iterator, Optional

from ..downloader import download_dataset
from ..utils import calculate_mask_ratio, create_black_mask_png, get_image_dimensions
from .base import BaseDatasetSource, DataRecord


class AutoSpliceSource(BaseDatasetSource):
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = (
            data_dir
            or os.environ.get("AUTOSPLICE_DIR")
            or os.path.join(os.getcwd(), "data", "AutoSplice")
            or os.path.join(os.getcwd(), "data", "autosplice")
        )

    @property
    def name(self) -> str:
        return "AutoSplice"

    def _ensure_data_available(self) -> bool:
        for cand in ["Forged_JPEG100", "Forged_JPEG90", "Forged", "forged", "Authentic", "authentic"]:
            if os.path.exists(os.path.join(self.data_dir, cand)):
                return True
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
        # AutoSplice layout: Forged_JPEG100/ (or Forged_JPEG90/), Authentic/, Mask/
        forged_dir = None
        for cand in ["Forged_JPEG100", "Forged_JPEG90", "Forged", "forged"]:
            p = os.path.join(self.data_dir, cand)
            if os.path.exists(p):
                forged_dir = p
                break

        auth_dir = None
        for cand in ["Authentic", "authentic", "Real", "real"]:
            p = os.path.join(self.data_dir, cand)
            if os.path.exists(p):
                auth_dir = p
                break

        mask_dir = None
        for cand in ["Mask", "mask", "masks"]:
            p = os.path.join(self.data_dir, cand)
            if os.path.exists(p):
                mask_dir = p
                break

        yielded_fake = 0
        yielded_real = 0

        # 1. Fake records
        if forged_dir and os.path.exists(forged_dir):
            for fname in sorted(os.listdir(forged_dir)):
                if yielded_fake >= target_fake:
                    break
                if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue

                img_path = os.path.join(forged_dir, fname)
                base_name, _ = os.path.splitext(fname)

                mask_path = None
                if mask_dir:
                    for cand_m in [f"{base_name}_mask.png", f"{base_name}.png", f"{base_name}.jpg"]:
                        fp = os.path.join(mask_dir, cand_m)
                        if os.path.exists(fp):
                            mask_path = fp
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
                        id=f"autosplice_{base_name}",
                        image_bytes=img_bytes,
                        image_path=fname,
                        mask_bytes=mask_bytes,
                        mask_path=os.path.basename(mask_path) if mask_path else f"{base_name}_mask.png",
                        label=1,
                        label_text="fake",
                        dataset="AutoSplice",
                        method="DALL-E2",
                        mask_ratio=ratio,
                    )
                    yielded_fake += 1
                except Exception:
                    continue

        # 2. Authentic records
        if auth_dir and os.path.exists(auth_dir):
            for fname in sorted(os.listdir(auth_dir)):
                if yielded_real >= target_real:
                    break
                if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue

                img_path = os.path.join(auth_dir, fname)
                base_name, _ = os.path.splitext(fname)
                try:
                    with open(img_path, "rb") as f:
                        img_bytes = f.read()
                    w, h = get_image_dimensions(img_bytes)
                    black_mask = create_black_mask_png(w, h)

                    yield DataRecord(
                        id=f"autosplice_real_{base_name}",
                        image_bytes=img_bytes,
                        image_path=fname,
                        mask_bytes=black_mask,
                        mask_path=f"real_{base_name}_mask.png",
                        label=0,
                        label_text="real",
                        dataset="AutoSplice",
                        method="authentic",
                        mask_ratio=0.0,
                    )
                    yielded_real += 1
                except Exception:
                    continue

        print(f"[{self.name}] Completed: {yielded_real} real, {yielded_fake} fake.")
