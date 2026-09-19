"""
Dataset source loader for BR-Gen (AAAI 2026, arXiv:2504.11922).
Parses the standard IMDL directory hierarchy:
  Au/  (Authentic images)
  Tp/  (Tampered images: {Method}_Stuff_{SourceID}.png)
  Gt/  (Ground truth masks: {Method}_Stuff_{SourceID}_mask.png)
"""

import os
from typing import Iterator, Optional

from ..utils import calculate_mask_ratio, create_black_mask_png, get_image_dimensions
from .base import BaseDatasetSource, DataRecord


class BRGenSource(BaseDatasetSource):
    def __init__(self, data_dir: Optional[str] = None):
        # Check explicit path, environment variable, or common project folders
        self.data_dir = (
            data_dir
            or os.environ.get("BRGEN_DIR")
            or os.path.join(os.getcwd(), "data", "BR-Gen")
            or os.path.join(os.getcwd(), "datasets", "BRGen_Stuff_Val")
        )

    @property
    def name(self) -> str:
        return "BR-Gen"

    def stream_records(
        self,
        target_real: int = 1000,
        target_fake: int = 1000,
    ) -> Iterator[DataRecord]:
        tp_dir = os.path.join(self.data_dir, "Tp")
        gt_dir = os.path.join(self.data_dir, "Gt")
        au_dir = os.path.join(self.data_dir, "Au")

        if not os.path.exists(tp_dir) or not os.path.exists(gt_dir):
            print(
                f"[{self.name}] Notice: Local directory '{self.data_dir}' not found.\n"
                f"  BR-Gen is distributed as a multi-part split archive ('BR-Gen_split.zip', '.z01'..'.z12')\n"
                f"  on Hugging Face (clpbc/BR-Gen) and Baidu Netdisk. To ingest full BR-Gen data, unpack into\n"
                f"  '{self.data_dir}' or pass --brgen-dir <path>."
            )
            return

        print(f"[{self.name}] Scanning directory {self.data_dir}...")
        tp_files = sorted(os.listdir(tp_dir))
        au_files = sorted(os.listdir(au_dir)) if os.path.exists(au_dir) else []

        yielded_fake = 0
        yielded_real = 0

        # 1. Process fake records
        for fname in tp_files:
            if yielded_fake >= target_fake:
                break
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            base_name, _ = os.path.splitext(fname)
            mask_fname = f"{base_name}_mask.png"
            mask_path = os.path.join(gt_dir, mask_fname)

            # Try alternate mask naming conventions if needed
            if not os.path.exists(mask_path):
                mask_fname = f"{base_name}.png"
                mask_path = os.path.join(gt_dir, mask_fname)
                if not os.path.exists(mask_path):
                    continue

            tp_path = os.path.join(tp_dir, fname)
            with open(tp_path, "rb") as f:
                img_bytes = f.read()
            with open(mask_path, "rb") as f:
                mask_bytes = f.read()

            ratio = calculate_mask_ratio(mask_bytes)
            # Avoid 100% fake and 0% fake
            if not (0.0005 <= ratio <= 0.95):
                continue

            # Extract method name from prefix (e.g. MAT_Stuff_COCO_... -> MAT, PowerPaint_Stuff_... -> PowerPaint)
            parts = base_name.split("_")
            method = parts[0] if len(parts) > 0 else "unknown_inpainting"

            yield DataRecord(
                id=f"brgen_{base_name}",
                image_bytes=img_bytes,
                image_path=fname,
                mask_bytes=mask_bytes,
                mask_path=mask_fname,
                label=1,
                label_text="fake",
                dataset="BR-Gen",
                method=method,
                mask_ratio=ratio,
            )
            yielded_fake += 1

        # 2. Process real records
        for fname in au_files:
            if yielded_real >= target_real:
                break
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            au_path = os.path.join(au_dir, fname)
            with open(au_path, "rb") as f:
                au_bytes = f.read()

            w, h = get_image_dimensions(au_bytes)
            black_mask = create_black_mask_png(w, h)
            base_name, _ = os.path.splitext(fname)

            yield DataRecord(
                id=f"brgen_real_{base_name}",
                image_bytes=au_bytes,
                image_path=fname,
                mask_bytes=black_mask,
                mask_path=f"real_{base_name}_mask.png",
                label=0,
                label_text="real",
                dataset="BR-Gen",
                method="authentic",
                mask_ratio=0.0,
            )
            yielded_real += 1

        print(f"[{self.name}] Completed: {yielded_real} real, {yielded_fake} fake.")
