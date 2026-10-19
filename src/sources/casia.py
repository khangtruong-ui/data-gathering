"""
Dataset source loader for CASIA V1.0 and CASIA V2.0 benchmarks.
Supports both pre-downloaded local folders and automated on-demand downloading.
Filters for localized forgeries and synthesizes black ground-truth masks for authentic images.
"""

import os
from typing import Iterator, Optional

from ..downloader import download_dataset
from ..utils import calculate_mask_ratio, create_black_mask_png, get_image_dimensions
from .base import BaseDatasetSource, DataRecord


class CasiaSource(BaseDatasetSource):
    def __init__(self, version: str = "v1", data_dir: Optional[str] = None):
        """
        version: 'v1' or 'v2'
        data_dir: optional local path to CASIA1 or CASIA2 directory
        """
        self.version = version.lower()
        self.dataset_key = "casia1" if self.version == "v1" else "casia2"
        self.data_dir = (
            data_dir
            or os.environ.get(f"CASIA_{self.version.upper()}_DIR")
            or os.path.join(os.getcwd(), "data", "CASIA", f"CASIA{self.version.replace('v', '')}")
            or os.path.join(os.getcwd(), "data", f"casia{self.version.replace('v', '')}")
        )

    @property
    def name(self) -> str:
        return f"CASIA_{self.version.upper()}"

    def _ensure_data_available(self) -> bool:
        if os.path.exists(self.data_dir) and len(os.listdir(self.data_dir)) > 0:
            return True
        try:
            print(f"[{self.name}] Data not found at '{self.data_dir}'. Triggering automated download...")
            downloaded_dir = download_dataset(self.dataset_key, target_dir="data")
            if os.path.exists(downloaded_dir):
                self.data_dir = downloaded_dir
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
        yielded_real = 0
        yielded_fake = 0

        # CASIA 1 directory layout: Au/ (authentic), Sp/ (spliced/fake), mask/
        # CASIA 2 layout: Au/, Tp/, and Groundtruth/ (or mask/)
        au_dir = os.path.join(self.data_dir, "Au")
        fake_dir = (
            os.path.join(self.data_dir, "fake")
            if os.path.exists(os.path.join(self.data_dir, "fake"))
            else (
                os.path.join(self.data_dir, "Sp")
                if os.path.exists(os.path.join(self.data_dir, "Sp"))
                else os.path.join(self.data_dir, "Tp")
            )
        )
        mask_dir = (
            os.path.join(self.data_dir, "mask")
            if os.path.exists(os.path.join(self.data_dir, "mask"))
            else os.path.join(self.data_dir, "CASIA 2 Groundtruth")
        )

        # 1. Stream fake records
        if os.path.exists(fake_dir):
            for root, _, files in os.walk(fake_dir):
                if yielded_fake >= target_fake:
                    break
                for fname in sorted(files):
                    if yielded_fake >= target_fake:
                        break
                    if not fname.lower().endswith((".jpg", ".png", ".tif", ".bmp")):
                        continue

                    img_path = os.path.join(root, fname)
                    base_name, _ = os.path.splitext(fname)

                    # Look for corresponding mask: e.g. {base_name}_gt.png or {base_name}.png
                    mask_path = None
                    if os.path.exists(mask_dir):
                        candidates = [
                            os.path.join(mask_dir, f"{base_name}_gt.png"),
                            os.path.join(mask_dir, f"{base_name}.png"),
                            os.path.join(mask_dir, f"{base_name}_mask.png"),
                            os.path.join(mask_dir, f"{base_name}.bmp"),
                        ]
                        for c in candidates:
                            if os.path.exists(c):
                                mask_path = c
                                break

                    try:
                        with open(img_path, "rb") as f:
                            img_bytes = f.read()
                        if mask_path and os.path.exists(mask_path):
                            with open(mask_path, "rb") as f:
                                mask_bytes = f.read()
                        else:
                            # Generate placeholder mask if mask file not present
                            w, h = get_image_dimensions(img_bytes)
                            mask_bytes = create_black_mask_png(w, h)

                        ratio = calculate_mask_ratio(mask_bytes)
                        # Avoid full generation / empty mask
                        if not (0.0005 <= ratio <= 0.95):
                            continue

                        yield DataRecord(
                            id=f"{self.name.lower()}_{base_name}",
                            image_bytes=img_bytes,
                            image_path=fname,
                            mask_bytes=mask_bytes,
                            mask_path=os.path.basename(mask_path) if mask_path else f"{base_name}_mask.png",
                            label=1,
                            label_text="fake",
                            dataset=self.name,
                            method="Splicing",
                            mask_ratio=ratio,
                        )
                        yielded_fake += 1
                    except Exception:
                        continue

        # 2. Stream authentic records
        if os.path.exists(au_dir):
            for root, _, files in os.walk(au_dir):
                if yielded_real >= target_real:
                    break
                for fname in sorted(files):
                    if yielded_real >= target_real:
                        break
                    if not fname.lower().endswith((".jpg", ".png", ".tif", ".bmp")):
                        continue

                    img_path = os.path.join(root, fname)
                    base_name, _ = os.path.splitext(fname)
                    try:
                        with open(img_path, "rb") as f:
                            img_bytes = f.read()
                        w, h = get_image_dimensions(img_bytes)
                        black_mask = create_black_mask_png(w, h)

                        yield DataRecord(
                            id=f"{self.name.lower()}_real_{base_name}",
                            image_bytes=img_bytes,
                            image_path=fname,
                            mask_bytes=black_mask,
                            mask_path=f"real_{base_name}_mask.png",
                            label=0,
                            label_text="real",
                            dataset=self.name,
                            method="authentic",
                            mask_ratio=0.0,
                        )
                        yielded_real += 1
                    except Exception:
                        continue

        print(f"[{self.name}] Completed: {yielded_real} real, {yielded_fake} fake.")
