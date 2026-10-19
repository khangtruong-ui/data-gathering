"""
Dataset source loader for COCO-Inpaint (streamed from KhangTruong/COCO-inpainted on HF Hub).
Filters for localized inpainting modifications and synthesizes black masks for authentic images.
"""

import os
import shutil
import tempfile
from typing import Iterator, List
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

from ..utils import calculate_mask_ratio, create_black_mask_png, get_image_dimensions
from .base import BaseDatasetSource, DataRecord


class CocoInpaintSource(BaseDatasetSource):
    REPO_ID = "KhangTruong/COCO-inpainted"

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "COCO_Inpaint"

    def _get_shards(self) -> List[str]:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            files = api.list_repo_files(repo_id=self.REPO_ID, repo_type="dataset")
            matched = sorted([f for f in files if f.startswith("data/") and f.endswith(".parquet")])
            if matched:
                return matched
        except Exception:
            pass
        return [f"data/train-{i:05d}.parquet" for i in range(4)]

    def stream_records(
        self,
        target_real: int = 1000,
        target_fake: int = 1000,
    ) -> Iterator[DataRecord]:
        shards = self._get_shards()
        print(f"[{self.name}] Streaming from {self.REPO_ID} ({len(shards)} shards)...")

        yielded_real = 0
        yielded_fake = 0

        for shard_rel in shards:
            if yielded_fake >= target_fake and yielded_real >= target_real:
                break

            tmp_dir = tempfile.mkdtemp(prefix="cocoinpaint_shard_")
            try:
                local_shard = hf_hub_download(
                    repo_id=self.REPO_ID,
                    filename=shard_rel,
                    repo_type="dataset",
                    local_dir=tmp_dir,
                )

                pf = pq.ParquetFile(local_shard)
                for rg_idx in range(pf.num_row_groups):
                    if yielded_fake >= target_fake and yielded_real >= target_real:
                        break

                    rg = pf.read_row_group(rg_idx)
                    for i in range(len(rg)):
                        if yielded_fake >= target_fake and yielded_real >= target_real:
                            break

                        img_dict = rg["image"][i].as_py()
                        mask_dict = rg["mask"][i].as_py()
                        img_bytes = img_dict.get("bytes") if img_dict else None
                        mask_bytes = mask_dict.get("bytes") if mask_dict else None

                        if not img_bytes:
                            continue

                        # Process fake
                        if mask_bytes and yielded_fake < target_fake:
                            ratio = calculate_mask_ratio(mask_bytes)
                            if 0.0005 <= ratio <= 0.95:
                                yield DataRecord(
                                    id=f"cocoinpaint_fake_{rg_idx}_{i}",
                                    image_bytes=img_bytes,
                                    image_path=f"cocoinpaint_{rg_idx}_{i}.jpg",
                                    mask_bytes=mask_bytes,
                                    mask_path=f"cocoinpaint_{rg_idx}_{i}_mask.png",
                                    label=1,
                                    label_text="fake",
                                    dataset="COCO_Inpaint",
                                    method="Inpainting",
                                    mask_ratio=ratio,
                                )
                                yielded_fake += 1

                        # Process authentic counterpart
                        if yielded_real < target_real and yielded_real < yielded_fake:
                            w, h = get_image_dimensions(img_bytes)
                            black_mask = create_black_mask_png(w, h)
                            yield DataRecord(
                                id=f"cocoinpaint_real_{rg_idx}_{i}",
                                image_bytes=img_bytes,
                                image_path=f"cocoinpaint_real_{rg_idx}_{i}.jpg",
                                mask_bytes=black_mask,
                                mask_path=f"cocoinpaint_real_{rg_idx}_{i}_mask.png",
                                label=0,
                                label_text="real",
                                dataset="COCO_Inpaint",
                                method="authentic",
                                mask_ratio=0.0,
                            )
                            yielded_real += 1
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        print(f"[{self.name}] Completed: {yielded_real} real, {yielded_fake} fake.")
