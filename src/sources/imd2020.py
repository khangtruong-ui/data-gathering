"""
Dataset stream reader for IMD2020 Inpainting (arXiv:1912.11038 / WACVW 2020).
Streams parquet shards directly from Hugging Face (KhangTruong/IMD2020).
Cleans up temporary shard downloads immediately to maintain safe disk usage.
"""

import os
import shutil
import tempfile
from typing import Iterator, List
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

from ..utils import calculate_mask_ratio, create_black_mask_png, get_image_dimensions
from .base import BaseDatasetSource, DataRecord


class IMD2020Source(BaseDatasetSource):
    REPO_ID = "KhangTruong/IMD2020"

    def __init__(self, split: str = "test"):
        """
        split: 'test', 'validation', or 'train'
        """
        self.split = split

    @property
    def name(self) -> str:
        return "IMD2020"

    def _get_shard_names(self) -> List[str]:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            files = api.list_repo_files(repo_id=self.REPO_ID, repo_type="dataset")
            matched = sorted(
                [f for f in files if f.startswith(f"data/{self.split}-") and f.endswith(".parquet")]
            )
            if matched:
                return matched
        except Exception as e:
            print(f"[{self.name}] Warning: HfApi listing failed ({e}), using default shards.")

        if self.split == "test":
            return [f"data/test-{i:05d}-of-00003.parquet" for i in range(3)]
        elif self.split == "validation":
            return [f"data/validation-{i:05d}-of-00003.parquet" for i in range(3)]
        return [f"data/train-{i:05d}-of-00021.parquet" for i in range(21)]

    def stream_records(
        self,
        target_real: int = 1000,
        target_fake: int = 1000,
    ) -> Iterator[DataRecord]:
        shards = self._get_shard_names()
        print(f"[{self.name}] Streaming from {self.REPO_ID} ({len(shards)} shards available in split '{self.split}')...")

        yielded_real = 0
        yielded_fake = 0

        for shard_rel_path in shards:
            if yielded_fake >= target_fake and yielded_real >= target_real:
                break

            tmp_dir = tempfile.mkdtemp(prefix="imd2020_shard_")
            try:
                local_shard_path = hf_hub_download(
                    repo_id=self.REPO_ID,
                    filename=shard_rel_path,
                    repo_type="dataset",
                    local_dir=tmp_dir,
                )

                pf = pq.ParquetFile(local_shard_path)
                for rg_idx in range(pf.num_row_groups):
                    if yielded_fake >= target_fake and yielded_real >= target_real:
                        break

                    rg = pf.read_row_group(rg_idx)
                    num_rows = len(rg)

                    for i in range(num_rows):
                        if yielded_fake >= target_fake and yielded_real >= target_real:
                            break

                        img_dict = rg["image"][i].as_py()
                        mask_dict = rg["mask"][i].as_py()

                        img_bytes = img_dict.get("bytes") if img_dict else None
                        mask_bytes = mask_dict.get("bytes") if mask_dict else None
                        img_path = img_dict.get("path") or f"imd2020_{rg_idx}_{i}.jpg"
                        mask_path = mask_dict.get("path") or f"imd2020_{rg_idx}_{i}_mask.png"

                        if not img_bytes:
                            continue

                        base_id = os.path.splitext(os.path.basename(img_path))[0]

                        # 1. Process fake (inpainting) sample
                        if yielded_fake < target_fake and mask_bytes:
                            ratio = calculate_mask_ratio(mask_bytes)
                            if 0.0005 <= ratio <= 0.95:
                                yield DataRecord(
                                    id=f"imd2020_{base_id}",
                                    image_bytes=img_bytes,
                                    image_path=os.path.basename(img_path),
                                    mask_bytes=mask_bytes,
                                    mask_path=os.path.basename(mask_path),
                                    label=1,
                                    label_text="fake",
                                    dataset="IMD2020",
                                    method="Inpainting",
                                    mask_ratio=ratio,
                                )
                                yielded_fake += 1

                        # 2. Process authentic counterpart (black mask)
                        if yielded_real < target_real and yielded_real < yielded_fake:
                            w, h = get_image_dimensions(img_bytes)
                            black_mask = create_black_mask_png(w, h)
                            yield DataRecord(
                                id=f"imd2020_real_{base_id}",
                                image_bytes=img_bytes,
                                image_path=os.path.basename(img_path),
                                mask_bytes=black_mask,
                                mask_path=f"real_{os.path.basename(mask_path)}",
                                label=0,
                                label_text="real",
                                dataset="IMD2020",
                                method="authentic",
                                mask_ratio=0.0,
                            )
                            yielded_real += 1

            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        print(f"[{self.name}] Completed streaming: {yielded_real} real, {yielded_fake} fake.")
