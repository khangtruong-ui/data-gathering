"""
Dataset builder: aggregates records from multiple sources, enforces balance
(e.g., 1000 real / 1000 fake per dataset), and writes optimized Parquet shards.

Features:
- Streaming batch-to-disk architecture to prevent RAM blowup (bounded memory footprint).
- Strict disk usage monitoring (configurable ceiling, default 100GB).
- Checkpointing and resumption support (saves progress in checkpoint.json).
- On-the-fly shard uploading and immediate deletion (--delete-after-upload).
- TQDM progress tracking for live streaming and shard writes.
"""

from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
import queue
import shutil
import tempfile
import threading
from typing import Callable, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .sources.base import BaseDatasetSource, DataRecord


class DatasetBuilder:
    def __init__(self, output_dir: str = "output", max_disk_gb: float = 100.0):
        self.output_dir = output_dir
        self.max_disk_gb = max_disk_gb
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

    def get_checkpoint_path(self) -> str:
        return os.path.join(self.output_dir, "checkpoint.json")

    def load_checkpoint(self) -> Optional[dict]:
        path = self.get_checkpoint_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Notice: Failed to load checkpoint ({e}). Starting fresh.")
        return None

    def save_checkpoint(self, checkpoint_data: dict):
        path = self.get_checkpoint_path()
        temp_path = f"{path}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2)
            os.replace(temp_path, path)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint ({e}).")

    def get_directory_size_bytes(self, dir_path: str) -> int:
        total = 0
        if not os.path.exists(dir_path):
            return 0
        for root, _, files in os.walk(dir_path):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        return total

    def check_disk_usage(self):
        """Ensures that local disk usage for outputs and temporary files does not exceed max_disk_gb."""
        output_bytes = self.get_directory_size_bytes(self.output_dir)
        output_gb = output_bytes / (1024**3)

        # Check overall filesystem free space as well
        try:
            usage = shutil.disk_usage(self.output_dir)
            free_gb = usage.free / (1024**3)
            if free_gb < 2.0:
                raise RuntimeError(
                    f"CRITICAL DISK WARNING: Low filesystem space! Only {free_gb:.2f} GB free."
                )
        except Exception as e:
            if "CRITICAL" in str(e):
                raise

        if output_gb >= self.max_disk_gb:
            raise RuntimeError(
                f"DISK LIMIT REACHED: Output directory size ({output_gb:.2f} GB) exceeded "
                f"maximum configured threshold ({self.max_disk_gb:.2f} GB). "
                f"Enable --delete-after-upload or increase --max-disk-gb."
            )

    def _write_shard(self, batch: List[DataRecord], shard_idx: int, split_name: str) -> dict:
        """Serializes a single batch into a compressed Parquet shard and clears memory."""
        self.check_disk_usage()

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

        shard_filename = f"{split_name}-{shard_idx:05d}.parquet"
        shard_path = os.path.join(self.output_dir, shard_filename)
        pq.write_table(table, shard_path, compression="snappy")

        # Compute per-dataset & per-method breakdown
        datasets_count: Dict[str, int] = {}
        methods_count: Dict[str, int] = {}
        for d in datasets:
            datasets_count[d] = datasets_count.get(d, 0) + 1
        for m in methods:
            methods_count[m] = methods_count.get(m, 0) + 1

        return {
            "filename": shard_filename,
            "path": shard_path,
            "num_rows": len(batch),
            "datasets": datasets_count,
            "methods": methods_count,
            "uploaded": False,
        }

    def build_split(
        self,
        sources: List[BaseDatasetSource],
        split_name: str = "cross_test",
        target_real_per_dataset: int = 1000,
        target_fake_per_dataset: int = 1000,
        batch_size: int = 500,
        max_workers: int = 4,
        resume: bool = True,
        uploader_callback: Optional[Callable[[str], None]] = None,
        delete_after_upload: bool = False,
    ) -> List[str]:
        """
        Iterates over all sources, streams records via a bounded-memory queue,
        and writes out Parquet shards incrementally to prevent RAM and disk blowup.
        """
        print(f"\n=======================================================")
        print(f"Building split '{split_name}' across {len(sources)} datasets")
        print(f"Target per dataset: {target_real_per_dataset} real, {target_fake_per_dataset} fake")
        print(f"Batch size: {batch_size} samples per Parquet shard")
        print(f"Max disk ceiling: {self.max_disk_gb} GB")
        print(f"Resume checkpoint: {resume}")
        print(f"Delete after upload: {delete_after_upload}")
        print(f"Concurrency: {max_workers} worker threads")
        print(f"=======================================================\n")

        # Load checkpoint if resuming
        checkpoint = self.load_checkpoint() if resume else None
        if not checkpoint:
            checkpoint = {
                "split_name": split_name,
                "target_real": target_real_per_dataset,
                "target_fake": target_fake_per_dataset,
                "completed_shards": [],
                "source_progress": {},
                "total_samples": 0,
            }

        prev_target_real = checkpoint.get("target_real", target_real_per_dataset)
        prev_target_fake = checkpoint.get("target_fake", target_fake_per_dataset)
        limits_increased = (
            target_real_per_dataset > prev_target_real
            or target_fake_per_dataset > prev_target_fake
        )
        checkpoint["target_real"] = target_real_per_dataset
        checkpoint["target_fake"] = target_fake_per_dataset

        completed_shards: List[dict] = checkpoint.get("completed_shards", [])
        source_progress: Dict[str, dict] = checkpoint.get("source_progress", {})
        shard_idx = len(completed_shards)

        print(f"Resuming with {shard_idx} completed shards, {checkpoint.get('total_samples', 0)} samples recorded.")

        # Determine remaining requirements per source
        sources_to_run = []
        for s in sources:
            prog = source_progress.get(s.name, {"real": 0, "fake": 0})
            rem_real = max(0, target_real_per_dataset - prog.get("real", 0))
            rem_fake = max(0, target_fake_per_dataset - prog.get("fake", 0))
            already_completed = prog.get("completed", False) and not limits_increased
            if (rem_real > 0 or rem_fake > 0) and not already_completed:
                sources_to_run.append((s, rem_real, rem_fake))
            else:
                print(f"  [{s.name}] Already completed ({prog.get('real', 0)} real, {prog.get('fake', 0)} fake). Skipping.")

        if not sources_to_run:
            print("All requested sources have met their target limits in checkpoint.")
            return [s["path"] for s in completed_shards if os.path.exists(s.get("path", ""))]

        # Bounded queue to strictly prevent unbounded RAM accumulation
        record_queue = queue.Queue(maxsize=batch_size * 2)
        done_sentinel = object()
        active_workers = len(sources_to_run)
        lock = threading.Lock()

        def _worker(source: BaseDatasetSource, needed_real: int, needed_fake: int):
            curr_real = source_progress.get(source.name, {}).get("real", 0)
            curr_fake = source_progress.get(source.name, {}).get("fake", 0)
            try:
                for record in source.stream_records(
                    target_real=needed_real,
                    target_fake=needed_fake,
                ):
                    record_queue.put(record)
                    with lock:
                        if record.label == 0:
                            curr_real += 1
                        else:
                            curr_fake += 1
                        source_progress[source.name] = {
                            "real": curr_real,
                            "fake": curr_fake,
                            "completed": False,
                        }
                with lock:
                    source_progress[source.name] = {
                        "real": curr_real,
                        "fake": curr_fake,
                        "completed": (curr_real > 0 or curr_fake > 0),
                    }
            except Exception as e:
                print(f"\n[Worker Error] {source.name}: {e}")
            finally:
                record_queue.put(done_sentinel)

        # Launch worker threads
        executor = ThreadPoolExecutor(max_workers=min(max_workers, len(sources_to_run)))
        for s, rem_real, rem_fake in sources_to_run:
            executor.submit(_worker, s, rem_real, rem_fake)

        current_batch: List[DataRecord] = []
        finished_workers = 0
        total_pbar_target = sum(r + f for _, r, f in sources_to_run)

        with tqdm(
            total=total_pbar_target,
            desc="Streaming & Writing Shards",
            unit="img",
            dynamic_ncols=True,
        ) as pbar:
            while finished_workers < active_workers or not record_queue.empty():
                try:
                    item = record_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if item is done_sentinel:
                    finished_workers += 1
                    continue

                current_batch.append(item)
                pbar.update(1)

                if len(current_batch) >= batch_size:
                    shard_info = self._write_shard(current_batch, shard_idx, split_name)
                    completed_shards.append(shard_info)
                    checkpoint["completed_shards"] = completed_shards
                    checkpoint["source_progress"] = source_progress
                    checkpoint["total_samples"] = sum(s["num_rows"] for s in completed_shards)
                    self.save_checkpoint(checkpoint)

                    shard_path = shard_info["path"]
                    shard_idx += 1
                    current_batch.clear()
                    gc.collect()

                    if uploader_callback:
                        uploader_callback(shard_path)
                        shard_info["uploaded"] = True
                        self.save_checkpoint(checkpoint)
                        if delete_after_upload and os.path.exists(shard_path):
                            try:
                                os.remove(shard_path)
                                print(f"  [Disk-Safe] Deleted uploaded shard: {shard_path}")
                            except Exception as e:
                                print(f"  Warning: could not delete {shard_path}: {e}")

        # Flush any remaining records
        if current_batch:
            shard_info = self._write_shard(current_batch, shard_idx, split_name)
            completed_shards.append(shard_info)
            checkpoint["completed_shards"] = completed_shards
            checkpoint["source_progress"] = source_progress
            checkpoint["total_samples"] = sum(s["num_rows"] for s in completed_shards)
            self.save_checkpoint(checkpoint)

            shard_path = shard_info["path"]
            shard_idx += 1
            current_batch.clear()
            gc.collect()

            if uploader_callback:
                uploader_callback(shard_path)
                shard_info["uploaded"] = True
                self.save_checkpoint(checkpoint)
                if delete_after_upload and os.path.exists(shard_path):
                    try:
                        os.remove(shard_path)
                        print(f"  [Disk-Safe] Deleted uploaded shard: {shard_path}")
                    except Exception as e:
                        print(f"  Warning: could not delete {shard_path}: {e}")

        executor.shutdown(wait=True)
        checkpoint["source_progress"] = source_progress
        self.save_checkpoint(checkpoint)
        total_samples = checkpoint.get("total_samples", 0)
        print(f"\nBuild complete. Total samples across all shards: {total_samples}")
        return [s["path"] for s in completed_shards]
