"""
Dataset stream reader for OpenSDID (OpenSDI arXiv:2503.19653).
Streams parquet shards one-by-one from Hugging Face, cleans up immediately,
and filters for strictly localized forgeries (avoiding 100% fake images).
Uses remote Parquet metadata inspection to download only relevant shards
and prioritizes dual-content shards to fulfill targets with minimal downloads.
"""

from concurrent.futures import ThreadPoolExecutor
import os
import shutil
import tempfile
from typing import Iterator, List, Tuple

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

from ..utils import calculate_mask_ratio, create_black_mask_png, get_image_dimensions
from .base import BaseDatasetSource, DataRecord


class OpenSDIDSource(BaseDatasetSource):
    REPO_ID = "nebula/OpenSDI_test"

    GENERATOR_PREFIX = {
        "Flux.1": "flux",
        "SD1.5": "sd15",
        "SDXL": "sdxl",
        "SD3": "sd3",
        "SD2.1": "sd2",
    }

    FALLBACK_SHARDS = {
        "Flux.1": [f"data/flux-{i:05d}-of-00007.parquet" for i in range(7)],
        "SD1.5": [f"data/sd15-{i:05d}-of-00019.parquet" for i in range(19)],
        "SDXL": [f"data/sdxl-{i:05d}-of-00009.parquet" for i in range(9)],
        "SD3": [f"data/sd3-{i:05d}-of-00010.parquet" for i in range(10)],
        "SD2.1": [f"data/sd2-{i:05d}-of-00007.parquet" for i in range(7)],
    }

    def __init__(self, target_generator: str = "Flux.1"):
        self.target_generator = target_generator

    @property
    def name(self) -> str:
        return f"OpenSDID_{self.target_generator.replace('.', '')}"

    def _get_shards(self) -> List[str]:
        prefix = self.GENERATOR_PREFIX.get(self.target_generator, "flux")
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            files = api.list_repo_files(repo_id=self.REPO_ID, repo_type="dataset")
            matched = sorted(
                [f for f in files if f.startswith(f"data/{prefix}-") and f.endswith(".parquet")]
            )
            if matched:
                return matched
        except Exception as e:
            print(f"[{self.name}] Notice: HfApi listing failed ({e}), using fallback shard list.")
        return self.FALLBACK_SHARDS.get(self.target_generator, self.FALLBACK_SHARDS["Flux.1"])

    def _inspect_remote_shard(self, shard_rel_path: str) -> Tuple[int, bool, bool]:
        """
        Quickly inspects remote Parquet metadata using HTTP range requests via fsspec
        to avoid downloading large (~400MB) shards that lack the needed labels/masks.
        Returns (masks_count, has_real, has_fake).
        """
        try:
            import fsspec

            url = f"https://huggingface.co/datasets/{self.REPO_ID}/resolve/main/{shard_rel_path}"
            fs = fsspec.filesystem("http")
            with fs.open(url, "rb") as f:
                pf = pq.ParquetFile(f)
                mask_col = None
                label_col = None
                for j in range(pf.metadata.num_columns):
                    col_name = pf.metadata.row_group(0).column(j).path_in_schema
                    if col_name == "mask.bytes":
                        mask_col = j
                    elif col_name == "label":
                        label_col = j

                total_masks = 0
                has_real = False
                has_fake = False
                for r in range(pf.num_row_groups):
                    rg = pf.metadata.row_group(r)
                    if mask_col is not None and rg.column(mask_col).statistics:
                        total_masks += rg.column(mask_col).statistics.num_values
                    if label_col is not None and rg.column(label_col).statistics:
                        stat = rg.column(label_col).statistics
                        if stat.min == 0 or stat.max == 0:
                            has_real = True
                        if stat.min == 1 or stat.max == 1:
                            has_fake = True
                return total_masks, has_real, has_fake
        except Exception:
            return None, True, True

    def stream_records(
        self,
        target_real: int = 1000,
        target_fake: int = 1000,
    ) -> Iterator[DataRecord]:
        shards = self._get_shards()

        # Inspect remote shards in parallel to determine mask and label distributions quickly
        with ThreadPoolExecutor(max_workers=min(len(shards), 8)) as executor:
            shard_info = list(
                executor.map(
                    lambda s: (s, *self._inspect_remote_shard(s)),
                    shards,
                )
            )

        def priority(item):
            _, masks, real, _ = item
            if masks is None:
                return 1
            if masks > 0 and real:
                return 0  # Dual content: delivers both fake and real
            if masks > 0:
                return 1  # Localized fakes
            if real:
                return 2  # Reals only
            return 3  # Unmasked global synthesis

        shard_info.sort(key=priority)

        yielded_real = 0
        yielded_fake = 0

        for shard_rel_path, masks_count, has_real, has_fake in shard_info:
            if yielded_real >= target_real and yielded_fake >= target_fake:
                break

            # If we already have enough fakes, and this shard has no reals: skip
            if yielded_fake >= target_fake and (has_real is False):
                continue
            # If we already have enough reals, and this shard has no masks: skip
            if yielded_real >= target_real and (masks_count == 0):
                continue
            # If this shard has neither masks nor reals (e.g., 100% synthetic fake without masks): skip
            if (masks_count == 0) and (has_real is False):
                continue

            print(f"[{self.name}] Streaming shard {shard_rel_path}...")
            tmp_dir = tempfile.mkdtemp(prefix="opensdid_shard_")
            try:
                local_shard_path = hf_hub_download(
                    repo_id=self.REPO_ID,
                    filename=shard_rel_path,
                    repo_type="dataset",
                    local_dir=tmp_dir,
                )

                pf = pq.ParquetFile(local_shard_path)
                for rg_idx in range(pf.num_row_groups):
                    if yielded_real >= target_real and yielded_fake >= target_fake:
                        break

                    rg = pf.read_row_group(rg_idx)
                    labels = rg["label"].to_pylist()

                    # Check if row group has any needed samples
                    has_needed_fake = (yielded_fake < target_fake) and (1 in labels)
                    has_needed_real = (yielded_real < target_real) and (0 in labels)
                    if not has_needed_fake and not has_needed_real:
                        continue

                    for i in range(len(rg)):
                        if yielded_real >= target_real and yielded_fake >= target_fake:
                            break

                        lbl = labels[i]
                        key = rg["key"][i].as_py()
                        img_dict = rg["image"][i].as_py()
                        mask_dict = rg["mask"][i].as_py()

                        img_bytes = img_dict.get("bytes") if img_dict else None
                        if not img_bytes:
                            continue

                        base_key = os.path.basename(key)
                        if not base_key.lower().endswith((".png", ".jpg", ".jpeg")):
                            base_key = f"{base_key}.png"

                        # Process fake
                        if lbl == 1 and yielded_fake < target_fake:
                            if not mask_dict or not mask_dict.get("bytes"):
                                continue
                            mask_bytes = mask_dict["bytes"]
                            ratio = calculate_mask_ratio(mask_bytes)

                            # Enforce localized tampering filter: avoid 100% fake and 0% fake
                            if not (0.0005 <= ratio <= 0.95):
                                continue

                            yield DataRecord(
                                id=f"opensdid_{self.target_generator}_{base_key}",
                                image_bytes=img_bytes,
                                image_path=base_key,
                                mask_bytes=mask_bytes,
                                mask_path=f"{os.path.splitext(base_key)[0]}_mask.png",
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
                                id=f"opensdid_real_{base_key}",
                                image_bytes=img_bytes,
                                image_path=base_key,
                                mask_bytes=black_mask,
                                mask_path=f"{os.path.splitext(base_key)[0]}_mask.png",
                                label=0,
                                label_text="real",
                                dataset=f"OpenSDID_{self.target_generator}",
                                method="authentic",
                                mask_ratio=0.0,
                            )
                            yielded_real += 1

            finally:
                # Always delete temporary shard immediately to preserve disk
                shutil.rmtree(tmp_dir, ignore_errors=True)

        print(
            f"[{self.name}] Completed streaming: {yielded_real} real, {yielded_fake} fake."
        )
