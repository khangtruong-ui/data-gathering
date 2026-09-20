"""
Dataset builder: aggregates records from multiple sources, enforces balance
(1000 real / 1000 fake per dataset), and writes optimized Parquet shards.
Includes TQDM progress bars for tracking real-time ingestion and serialization.
Supports concurrent dataset streaming via ThreadPoolExecutor.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .sources.base import BaseDatasetSource, DataRecord


class DatasetBuilder:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.schema = pa.schema(
            [
                ("id", pa.string()),
                (
                    "image",
                    pa.struct(
                        [
                            ("bytes", pa.binary()),
                            ("path", pa.string()),
                        ]
                    ),
                ),
                (
                    "mask",
                    pa.struct(
                        [
                            ("bytes", pa.binary()),
                            ("path", pa.string()),
                        ]
                    ),
                ),
                ("label", pa.int64()),
                ("label_text", pa.string()),
                ("dataset", pa.string()),
                ("method", pa.string()),
                ("mask_ratio", pa.float32()),
            ]
        )

    def build_split(
        self,
        sources: List[BaseDatasetSource],
        split_name: str = "cross_test",
        target_real_per_dataset: int = 1000,
        target_fake_per_dataset: int = 1000,
        batch_size: int = 500,
        max_workers: int = 4,
    ) -> List[str]:
        """
        Iterates over all sources, collects target records, and writes to Parquet shards.
        Uses ThreadPoolExecutor for concurrent network IO when max_workers > 1.
        Returns list of generated shard filepaths.
        """
        print(f"\n=======================================================")
        print(f"Building split '{split_name}' across {len(sources)} datasets")
        print(
            f"Target per dataset: {target_real_per_dataset} real, {target_fake_per_dataset} fake"
        )
        print(f"Concurrency: {max_workers} worker threads")
        print(f"=======================================================\n")

        all_records: List[DataRecord] = []

        def _stream_source(pos: int, source: BaseDatasetSource) -> List[DataRecord]:
            source_real = 0
            source_fake = 0
            total_target = target_real_per_dataset + target_fake_per_dataset
            records: List[DataRecord] = []

            with tqdm(
                total=total_target,
                desc=f"[{source.name}]",
                unit="img",
                position=pos,
                leave=True,
                dynamic_ncols=True,
            ) as pbar:
                for record in source.stream_records(
                    target_real=target_real_per_dataset,
                    target_fake=target_fake_per_dataset,
                ):
                    records.append(record)
                    if record.label == 0:
                        source_real += 1
                    else:
                        source_fake += 1
                    pbar.update(1)
                    pbar.set_postfix({"real": source_real, "fake": source_fake})
            return records

        if max_workers and max_workers > 1 and len(sources) > 1:
            print(
                f"Streaming {len(sources)} sources concurrently using ThreadPoolExecutor(max_workers={max_workers})...\n"
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(_stream_source, idx, source): idx
                    for idx, source in enumerate(sources)
                }
                results_by_idx = {}
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results_by_idx[idx] = future.result()

                # Print blank lines so subsequent output doesn't overwrite tqdm rows
                print("\n" * (len(sources) + 1))
                for idx in sorted(results_by_idx.keys()):
                    all_records.extend(results_by_idx[idx])
        else:
            for idx, source in enumerate(sources):
                all_records.extend(_stream_source(idx, source))

        total_samples = len(all_records)
        print(f"\nTotal collected records across all datasets: {total_samples}")

        if total_samples == 0:
            print("Warning: No records collected.")
            return []

        # Write out to Parquet shards with progress bar
        shards = []
        num_shards = max(1, (total_samples + batch_size - 1) // batch_size)

        with tqdm(
            total=num_shards, desc="Writing Parquet shards", unit="shard", dynamic_ncols=True
        ) as pbar:
            for shard_idx in range(num_shards):
                start = shard_idx * batch_size
                end = min(total_samples, start + batch_size)
                batch = all_records[start:end]

                ids = [r.id for r in batch]
                images = [{"bytes": r.image_bytes, "path": r.image_path} for r in batch]
                masks = [{"bytes": r.mask_bytes, "path": r.mask_path} for r in batch]
                labels = [r.label for r in batch]
                label_texts = [r.label_text for r in batch]
                datasets = [r.dataset for r in batch]
                methods = [r.method for r in batch]
                mask_ratios = [r.mask_ratio for r in batch]

                table = pa.Table.from_arrays(
                    [
                        pa.array(ids, type=pa.string()),
                        pa.array(images, type=self.schema.field("image").type),
                        pa.array(masks, type=self.schema.field("mask").type),
                        pa.array(labels, type=pa.int64()),
                        pa.array(label_texts, type=pa.string()),
                        pa.array(datasets, type=pa.string()),
                        pa.array(methods, type=pa.string()),
                        pa.array(mask_ratios, type=pa.float32()),
                    ],
                    schema=self.schema,
                )

                shard_filename = f"{split_name}-{shard_idx:05d}-of-{num_shards:05d}.parquet"
                shard_path = os.path.join(self.output_dir, shard_filename)
                pq.write_table(table, shard_path, compression="snappy")
                shards.append(shard_path)
                pbar.update(1)

        return shards
