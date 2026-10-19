"""
Dataset source loader for HiFi-IFDL (Hierarchical Fine-Grained Image Forgery Detection and Localization).
Guo et al., CVPR 2023 / IJCV 2024.
"""

import os
from typing import Iterator, Optional

from ..downloader import download_dataset
from ..utils import calculate_mask_ratio, create_black_mask_png, get_image_dimensions
from .base import BaseDatasetSource, DataRecord


class HiFiIFDLSource(BaseDatasetSource):
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = (
            data_dir
            or os.environ.get("HIFI_IFDL_DIR")
            or os.path.join(os.getcwd(), "data", "HiFi_IFDL")
            or os.path.join(os.getcwd(), "data", "hifi_ifdl")
        )

    @property
    def name(self) -> str:
        return "HiFi_IFDL"

    def _ensure_data_available(self) -> bool:
        if os.path.exists(self.data_dir) and len(os.listdir(self.data_dir)) > 0:
            return True
        try:
            print(f"[{self.name}] Data not found at '{self.data_dir}'. Triggering automated download...")
            downloaded = download_dataset("hifi_ifdl", target_dir="data")
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
        yielded_fake = 0
        yielded_real = 0

        # Scan for images and masks in HiFi folder
        img_candidates = []
        for root, _, files in os.walk(self.data_dir):
            if "mask" in root.lower():
                continue
            for f in sorted(files):
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_candidates.append(os.path.join(root, f))

        for img_path in img_candidates:
            if yielded_fake >= target_fake and yielded_real >= target_real:
                break

            fname = os.path.basename(img_path)
            base_name, _ = os.path.splitext(fname)

            # Search for corresponding mask
            mask_path = None
            parent = os.path.dirname(img_path)
            mask_dirs = [
                os.path.join(parent, "mask"),
                os.path.join(os.path.dirname(parent), "mask"),
                os.path.join(self.data_dir, "mask"),
            ]
            for md in mask_dirs:
                if os.path.exists(md):
                    for cand in [f"{base_name}_mask.png", f"{base_name}.png", f"{base_name}_gt.png"]:
                        fp = os.path.join(md, cand)
                        if os.path.exists(fp):
                            mask_path = fp
                            break
                if mask_path:
                    break

            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()

                if mask_path and os.path.exists(mask_path) and yielded_fake < target_fake:
                    with open(mask_path, "rb") as f:
                        mask_bytes = f.read()

                    ratio = calculate_mask_ratio(mask_bytes)
                    if 0.0005 <= ratio <= 0.95:
                        yield DataRecord(
                            id=f"hifi_{base_name}",
                            image_bytes=img_bytes,
                            image_path=fname,
                            mask_bytes=mask_bytes,
                            mask_path=os.path.basename(mask_path),
                            label=1,
                            label_text="fake",
                            dataset="HiFi_IFDL",
                            method="Hierarchical_Forgery",
                            mask_ratio=ratio,
                        )
                        yielded_fake += 1

                elif yielded_real < target_real and yielded_real < yielded_fake:
                    w, h = get_image_dimensions(img_bytes)
                    black_mask = create_black_mask_png(w, h)

                    yield DataRecord(
                        id=f"hifi_real_{base_name}",
                        image_bytes=img_bytes,
                        image_path=fname,
                        mask_bytes=black_mask,
                        mask_path=f"real_{base_name}_mask.png",
                        label=0,
                        label_text="real",
                        dataset="HiFi_IFDL",
                        method="authentic",
                        mask_ratio=0.0,
                    )
                    yielded_real += 1
            except Exception:
                continue

        print(f"[{self.name}] Completed: {yielded_real} real, {yielded_fake} fake.")
