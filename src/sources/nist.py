"""
Dataset source loader for NIST16 (Nimble Challenge 2016 / OpenMFC evaluation dataset).
"""

import os
from typing import Iterator, Optional

from ..downloader import download_dataset
from ..utils import calculate_mask_ratio, create_black_mask_png, get_image_dimensions
from .base import BaseDatasetSource, DataRecord


class NIST16Source(BaseDatasetSource):
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = (
            data_dir
            or os.environ.get("NIST16_DIR")
            or os.path.join(os.getcwd(), "data", "NIST16")
            or os.path.join(os.getcwd(), "data", "nist16")
        )

    @property
    def name(self) -> str:
        return "NIST16"

    def _ensure_data_available(self) -> bool:
        if os.path.exists(self.data_dir) and len(os.listdir(self.data_dir)) > 0:
            return True
        try:
            print(f"[{self.name}] Triggering automated download via standard benchmark bundle...")
            downloaded = download_dataset("nist16", target_dir="data")
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

        # NIST16 typically has 'alllist.txt' with pairs of 'image mask' or folders
        list_file = os.path.join(self.data_dir, "alllist.txt")
        pairs = []

        if os.path.exists(list_file):
            with open(list_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        pairs.append((os.path.join(self.data_dir, parts[0]), os.path.join(self.data_dir, parts[1])))
        else:
            # Fallback file traversal
            for root, _, files in os.walk(self.data_dir):
                for f in files:
                    if f.lower().endswith((".jpg", ".png", ".tif")) and not "mask" in f.lower():
                        base = os.path.splitext(f)[0]
                        mask_cand = os.path.join(root, f"{base}_mask.png")
                        if os.path.exists(mask_cand):
                            pairs.append((os.path.join(root, f), mask_cand))

        for img_path, mask_path in pairs:
            if yielded_fake >= target_fake and yielded_real >= target_real:
                break
            if not os.path.exists(img_path):
                continue

            fname = os.path.basename(img_path)
            base_name, _ = os.path.splitext(fname)

            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()

                # Process fake
                if os.path.exists(mask_path) and yielded_fake < target_fake:
                    with open(mask_path, "rb") as f:
                        mask_bytes = f.read()

                    ratio = calculate_mask_ratio(mask_bytes)
                    if 0.0005 <= ratio <= 0.95:
                        # Determine method (splice, removal, manipulation)
                        m_lower = mask_path.lower()
                        method = "Splicing" if "splice" in m_lower else ("Removal" if "remove" in m_lower else "Manipulation")

                        yield DataRecord(
                            id=f"nist16_{base_name}",
                            image_bytes=img_bytes,
                            image_path=fname,
                            mask_bytes=mask_bytes,
                            mask_path=os.path.basename(mask_path),
                            label=1,
                            label_text="fake",
                            dataset="NIST16",
                            method=method,
                            mask_ratio=ratio,
                        )
                        yielded_fake += 1

                # Process real
                if yielded_real < target_real and yielded_real < yielded_fake:
                    w, h = get_image_dimensions(img_bytes)
                    black_mask = create_black_mask_png(w, h)

                    yield DataRecord(
                        id=f"nist16_real_{base_name}",
                        image_bytes=img_bytes,
                        image_path=fname,
                        mask_bytes=black_mask,
                        mask_path=f"real_{base_name}_mask.png",
                        label=0,
                        label_text="real",
                        dataset="NIST16",
                        method="authentic",
                        mask_ratio=0.0,
                    )
                    yielded_real += 1
            except Exception:
                continue

        print(f"[{self.name}] Completed: {yielded_real} real, {yielded_fake} fake.")
