"""
Dataset stream reader for OpenSDID (OpenSDI arXiv:2503.19653).
Streams parquet shards one-by-one from Hugging Face, cleans up immediately,
and filters for strictly localized forgeries (avoiding 100% fake images).
"""

import os
import shutil
import tempfile
from typing import Iterator

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

from ..utils import calculate_mask_ratio, create_black_mask_png, get_image_dimensions
from .base import BaseDatasetSource, DataRecord


class OpenSDIDSource(BaseDatasetSource):
    REPO_ID = "nebula/OpenSDI_test"

    # Mapping of generator prefix to standard method name and sample parquet file
    GENERATOR_SHARDS = {
        "Flux.1": "data/flux-00000-of-00007.parquet",
        "SD1.5": "data/sd15-00000-of-00019.parquet",
        "SDXL": "data/sdxl-00000-of-00007.parquet",
        "SD3": "data/sd3-00000-of-00007.parquet",
        "SD2.1": "data/sd2-00000-of-00007.parquet",
    }

    def __init__(self, target_generator: str = "Flux.1"):
        self.target_generator = target_generator

    @property
    def name(self) -> str:
        return f"OpenSDID_{self.target_generator.replace('.', '')}"

    def stream_records(
        self,
        target_real: int = 1000,
        target_fake: int = 1000,
    ) -> Iterator[DataRecord]:
        shard_rel_path = self.GENERATOR_SHARDS.get(
            self.target_generator, "data/flux-00000-of-00007.parquet"
        )
        print(
            f"[{self.name}] Streaming {shard_rel_path} for generator {self.target_generator}..."
        )

        tmp_dir = tempfile.mkdtemp(prefix="opensdid_shard_")
        local_shard_path = None
        try:
            local_shard_path = hf_hub_download(
                repo_id=self.REPO_ID,
                filename=shard_rel_path,
                repo_type="dataset",
                local_dir=tmp_dir,
            )

            pf = pq.ParquetFile(local_shard_path)
            yielded_real = 0
            yielded_fake = 0

            for rg_idx in range(pf.num_row_groups):
                if yielded_real >= target_real and yielded_fake >= target_fake:
                    break

                rg = pf.read_row_group(rg_idx)
                for i in range(len(rg)):
                    if yielded_real >= target_real and yielded_fake >= target_fake:
                        break

                    lbl = rg["label"][i].as_py()
                    key = rg["key"][i].as_py()
                    img_dict = rg["image"][i].as_py()
                    mask_dict = rg["mask"][i].as_py()

                    img_bytes = img_dict.get("bytes") if img_dict else None
                    if not img_bytes:
                        continue

                    # Process fake
                    if lbl == 1 and yielded_fake < target_fake:
                        if not mask_dict or not mask_dict.get("bytes"):
                            # Skip full-synthesis / 100% fake images without localized mask
                            continue
                        mask_bytes = mask_dict["bytes"]
                        ratio = calculate_mask_ratio(mask_bytes)

                        # Enforce localized tampering filter: avoid 100% fake and 0% fake
                        if not (0.0005 <= ratio <= 0.95):
                            continue

                        yield DataRecord(
                            id=f"opensdid_{self.target_generator}_{key}",
                            image_bytes=img_bytes,
                            image_path=f"{key}.png",
                            mask_bytes=mask_bytes,
                            mask_path=f"{key}_mask.png",
                            label=1,
                            label_text="fake",
                            dataset=f"OpenSDID_{self.target_generator}",
                            method=self.target_generator,
                            mask_ratio=ratio,
                        )
                        yielded_fake += 1

                    # Process real
                    elif lbl == 0 and yielded_real < target_real:
                        w, h = get_image_dimensions(img_bytes)
                        black_mask = create_black_mask_png(w, h)

                        yield DataRecord(
                            id=f"opensdid_real_{key}",
                            image_bytes=img_bytes,
                            image_path=f"{key}.png",
                            mask_bytes=black_mask,
                            mask_path=f"{key}_mask.png",
                            label=0,
                            label_text="real",
                            dataset=f"OpenSDID_{self.target_generator}",
                            method="authentic",
                            mask_ratio=0.0,
                        )
                        yielded_real += 1

            print(
                f"[{self.name}] Completed streaming: {yielded_real} real, {yielded_fake} fake."
            )

        finally:
            # Always delete temporary shard immediately to preserve disk
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"[{self.name}] Cleaned up temporary shard to protect disk space.")
