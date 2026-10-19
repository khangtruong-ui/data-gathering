"""
Dataset source loader for IMD2020 Real-Life (2,010 internet-manipulated images with manual masks).
Novozámský et al., WACVW 2020.
"""

import os
from typing import Iterator, Optional

from ..downloader import download_dataset
from ..utils import calculate_mask_ratio, create_black_mask_png, get_image_dimensions
from .base import BaseDatasetSource, DataRecord


class IMD2020RealLifeSource(BaseDatasetSource):
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = (
            data_dir
            or os.environ.get("IMD2020_REALLIFE_DIR")
            or os.path.join(os.getcwd(), "data", "IMD2020")
            or os.path.join(os.getcwd(), "data", "imd2020_reallife")
        )

    @property
    def name(self) -> str:
        return "IMD2020_RealLife"

    def _ensure_data_available(self) -> bool:
        if os.path.exists(self.data_dir) and len(os.listdir(self.data_dir)) > 0:
            return True
        try:
            print(f"[{self.name}] Triggering automated download via standard benchmark bundle...")
            downloaded = download_dataset("imd2020_reallife", target_dir="data")
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
            os.path.join(self.data_dir, "fake_img")
            if os.path.exists(os.path.join(self.data_dir, "fake_img"))
            else (
                os.path.join(self.data_dir, "fake")
                if os.path.exists(os.path.join(self.data_dir, "fake"))
                else self.data_dir
            )
        )
        mask_dir = os.path.join(self.data_dir, "mask")
        orig_dir = (
            os.path.join(self.data_dir, "orig")
            if os.path.exists(os.path.join(self.data_dir, "orig"))
            else os.path.join(self.data_dir, "pristine")
        )

        yielded_fake = 0
        yielded_real = 0

        if not os.path.exists(img_dir):
            return

        for fname in sorted(os.listdir(img_dir)):
            if yielded_fake >= target_fake and yielded_real >= target_real:
                break
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(img_dir, fname)
            base_name, _ = os.path.splitext(fname)

            # Mask candidate
            mask_path = None
            if os.path.exists(mask_dir):
                for cand_name in [f"{base_name}_mask.png", f"{base_name}.png", f"{base_name}_gt.png"]:
                    cand_fp = os.path.join(mask_dir, cand_name)
                    if os.path.exists(cand_fp):
                        mask_path = cand_fp
                        break

            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()

                # Process fake
                if mask_path and os.path.exists(mask_path) and yielded_fake < target_fake:
                    with open(mask_path, "rb") as f:
                        mask_bytes = f.read()

                    ratio = calculate_mask_ratio(mask_bytes)
                    if 0.0005 <= ratio <= 0.95:
                        yield DataRecord(
                            id=f"imd2020_real_{base_name}",
                            image_bytes=img_bytes,
                            image_path=fname,
                            mask_bytes=mask_bytes,
                            mask_path=os.path.basename(mask_path),
                            label=1,
                            label_text="fake",
                            dataset="IMD2020_RealLife",
                            method="RealLife_Manipulation",
                            mask_ratio=ratio,
                        )
                        yielded_fake += 1

                # Process authentic counterpart
                if yielded_real < target_real and os.path.exists(orig_dir):
                    orig_cand = os.path.join(orig_dir, fname)
                    if os.path.exists(orig_cand):
                        with open(orig_cand, "rb") as f:
                            orig_bytes = f.read()
                        w, h = get_image_dimensions(orig_bytes)
                        black_mask = create_black_mask_png(w, h)

                        yield DataRecord(
                            id=f"imd2020_auth_{base_name}",
                            image_bytes=orig_bytes,
                            image_path=fname,
                            mask_bytes=black_mask,
                            mask_path=f"real_{base_name}_mask.png",
                            label=0,
                            label_text="real",
                            dataset="IMD2020_RealLife",
                            method="authentic",
                            mask_ratio=0.0,
                        )
                        yielded_real += 1
            except Exception:
                continue

        print(f"[{self.name}] Completed: {yielded_real} real, {yielded_fake} fake.")
