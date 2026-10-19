#!/usr/bin/env python3
"""
Main execution script for building, managing disk/RAM safety, and uploading
the merged AI tampering and manipulation localization dataset.

Usage:
  # 1. Quick Mock Test (verifies pipeline, schema, and Parquet writing with synthetic records):
  python3 build_merged_dataset.py --mode mock --limit 10

  # 2. Mock Test with Checkpoint Resume and Upload to KhangTruong/Tampered-Images-Combined-Dataset-2:
  python3 build_merged_dataset.py --mode mock --limit 10 --upload --resume

  # 3. Live Stream with IMD2020 (HF Hub) and CocoGlide (Disk-Safe):
  python3 build_merged_dataset.py --mode live --limit 5 --datasets imd2020 cocoglide --upload --delete-after-upload

  # 4. Synchronize remote Hugging Face README documentation directly to match existing shards:
  python3 build_merged_dataset.py --sync-doc
"""

import argparse
import collections
import os
import sys
from typing import List

from src.builder import DatasetBuilder
from src.sources import (
    AutoSpliceSource,
    BaseDatasetSource,
    BRGenSource,
    CasiaSource,
    CocoGlideSource,
    CocoInpaintSource,
    ColumbiaSource,
    CoverageSource,
    HiFiIFDLSource,
    IMD2020RealLifeSource,
    IMD2020Source,
    MockDatasetSource,
    NIST16Source,
    OpenSDIDSource,
    SAGIDSource,
    TGIFSource,
)
from src.uploader import HubUploader



def parse_args():
    parser = argparse.ArgumentParser(
        description="Build and upload multi-generator AI tampering dataset to KhangTruong/Tampered-Images-Combined-Dataset-2"
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        default="mock",
        help="Run mode: 'mock' uses synthetic procedural records; 'live' streams from real datasets",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Target number of real and fake images per dataset (default: 5 for mock, 1000 for full)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="cross_test",
        help="Hugging Face split name (default: cross_test)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="KhangTruong/Tampered-Images-Combined-Dataset-2",
        help="Target Hugging Face dataset repository",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Whether to upload the generated shards and updated README to Hugging Face",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dist_data",
        help="Directory to save generated Parquet files and checkpoint.json",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional list of specific datasets to include (e.g. imd2020 cocoglide sagid opensdid_flux opensdid_sd15 tgif brgen)",
    )
    parser.add_argument(
        "--brgen-dir",
        type=str,
        default=None,
        help="Optional local path to unpacked BR-Gen directory (containing Au/, Tp/, Gt/)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent worker threads for streaming datasets (default: 4)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume streaming from existing checkpoint.json if available (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Disable checkpoint resumption and start build from scratch",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of records per Parquet shard batch to prevent memory blowup (default: 500)",
    )
    parser.add_argument(
        "--max-disk-gb",
        type=float,
        default=100.0,
        help="Maximum allowable disk space threshold in GB before halting (default: 100.0)",
    )
    parser.add_argument(
        "--delete-after-upload",
        action="store_true",
        help="Delete local Parquet shard files immediately after uploading to Hugging Face to save disk space",
    )
    parser.add_argument(
        "--sync-doc",
        action="store_true",
        help="Inspect remote Hugging Face repository, clean up orphaned shards, and update README card directly",
    )
    return parser.parse_args()


def get_mock_sources(limit: int) -> List[BaseDatasetSource]:
    return [
        MockDatasetSource("OpenSDID_Flux", method="Flux.1"),
        MockDatasetSource("OpenSDID_SD15", method="SD1.5"),
        MockDatasetSource("OpenSDID_SDXL", method="SDXL"),
        MockDatasetSource("CocoGlide", method="GLIDE"),
        MockDatasetSource("TGIF2_SD2", method="SD2_spliced"),
        MockDatasetSource("SAGI-D", method="brushnet_epicrealism"),
        MockDatasetSource("IMD2020", method="Inpainting"),
        MockDatasetSource("IMD2020_RealLife", method="RealLife_Manipulation"),
        MockDatasetSource("CASIA_V1", method="Splicing"),
        MockDatasetSource("CASIA_V2", method="Splicing"),
        MockDatasetSource("Columbia_Gray", method="Splicing"),
        MockDatasetSource("Coverage", method="CopyMove"),
        MockDatasetSource("NIST16", method="Splicing"),
        MockDatasetSource("COCO_Inpaint", method="Inpainting"),
        MockDatasetSource("AutoSplice", method="DALL-E2"),
        MockDatasetSource("HiFi_IFDL", method="Hierarchical_Forgery"),
        MockDatasetSource("BR-Gen", method="PowerPaint"),
    ]


def get_live_sources(filter_names: List[str] = None, brgen_dir: str = None) -> List[BaseDatasetSource]:
    all_sources = [
        CocoGlideSource(),
        IMD2020Source(split="test"),
        IMD2020RealLifeSource(),
        CasiaSource(version="v1"),
        CasiaSource(version="v2"),
        ColumbiaSource(),
        CoverageSource(),
        NIST16Source(),
        CocoInpaintSource(),
        AutoSpliceSource(),
        HiFiIFDLSource(),
        SAGIDSource(),
        OpenSDIDSource(target_generator="Flux.1"),
        OpenSDIDSource(target_generator="SD1.5"),
        OpenSDIDSource(target_generator="SDXL"),
        TGIFSource(target_model="flux1dev"),
        BRGenSource(data_dir=brgen_dir),
    ]
    if not filter_names:
        return all_sources

    filter_lower = [f.lower().replace("-", "").replace("_", "") for f in filter_names]
    selected = [
        s for s in all_sources
        if any(k in s.name.lower().replace("-", "").replace("_", "") for k in filter_lower)
    ]
    return selected if selected else all_sources



def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("AI IMAGE TAMPERING DATASET MERGER & BUILDER")
    print("=" * 60)
    print(f"Target repo:         {args.repo}")
    print(f"Target split:        {args.split}")
    print(f"Output directory:    {args.output_dir}")
    print(f"Max disk ceiling:    {args.max_disk_gb} GB")
    print(f"Resume checkpoint:   {args.resume}")
    print(f"Delete after upload: {args.delete_after_upload}")

    # Synchronize documentation directly if requested
    if args.sync_doc:
        print("\nMode: Synchronize remote Hugging Face documentation only.")
        uploader = HubUploader(repo_id=args.repo)
        uploader.sync_hf_repo_documentation(split_name=args.split, remove_orphaned_shards=True)
        return

    print(f"Mode:                {args.mode}")
    print(f"Limit/source:        {args.limit} real, {args.limit} fake")
    print(f"Workers:             {args.workers}")
    print(f"Upload:              {args.upload}")
    print("=" * 60 + "\n")

    # 1. Setup sources
    if args.mode == "mock":
        sources = get_mock_sources(args.limit)
    else:
        sources = get_live_sources(args.datasets, brgen_dir=args.brgen_dir)

    print(f"Configured sources ({len(sources)}):")
    for s in sources:
        print(f"  - {s.name}")

    uploader = HubUploader(repo_id=args.repo) if args.upload else None

    # Setup uploader callback if continuous uploading is desired
    uploader_callback = None
    if args.upload:
        def _on_shard_written(shard_path: str):
            print(f"  [Auto-Upload] Committing shard {os.path.basename(shard_path)} to {args.repo}...")
            uploader.upload_single_shard(shard_path, split_name=args.split)
        uploader_callback = _on_shard_written

    # 2. Build dataset
    builder = DatasetBuilder(output_dir=args.output_dir, max_disk_gb=args.max_disk_gb)
    shard_paths = builder.build_split(
        sources=sources,
        split_name=args.split,
        target_real_per_dataset=args.limit,
        target_fake_per_dataset=args.limit,
        batch_size=args.batch_size,
        max_workers=args.workers,
        resume=args.resume,
        uploader_callback=uploader_callback,
        delete_after_upload=args.delete_after_upload,
    )

    if not shard_paths:
        print("Notice: No new shards generated.")

    # 3. Compute verified statistics from checkpoint / shards
    checkpoint = builder.load_checkpoint()
    datasets_distribution = collections.Counter()
    methods_distribution = collections.Counter()
    total_samples = 0

    if checkpoint and "completed_shards" in checkpoint:
        for s in checkpoint["completed_shards"]:
            total_samples += s.get("num_rows", 0)
            datasets_distribution.update(s.get("datasets", {}))
            methods_distribution.update(s.get("methods", {}))
    elif shard_paths:
        import pyarrow.parquet as pq
        for p in shard_paths:
            if os.path.exists(p):
                t = pq.read_table(p, columns=["dataset", "method"])
                total_samples += len(t)
                datasets_distribution.update(t["dataset"].to_pylist())
                methods_distribution.update(t["method"].to_pylist())

    print(f"\nFinal Verified Ingestion Summary ({total_samples:,} Total Samples):")
    for d, c in sorted(datasets_distribution.items()):
        print(f"  - Dataset {d}: {c:,} samples")
    for m, c in sorted(methods_distribution.items()):
        print(f"  - Method {m}: {c:,} samples")

    # 4. Finalize README update if upload enabled
    if args.upload and uploader:
        print("\nUpdating remote dataset card on Hugging Face...")
        uploader.update_dataset_card(
            split_name=args.split,
            total_samples=total_samples,
            datasets_distribution=dict(datasets_distribution),
            methods_distribution=dict(methods_distribution),
        )
        print(f"Upload and document synchronization complete: https://huggingface.co/datasets/{args.repo}")
    else:
        print("\nSkipping upload (use --upload to push to Hugging Face).")
        print(f"Local shards and checkpoint available in: {args.output_dir}/")


if __name__ == "__main__":
    main()
