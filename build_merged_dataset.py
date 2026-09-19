#!/usr/bin/env python3
"""
Main execution script for building and uploading the merged AI tampering dataset.

Usage:
  # 1. Quick Mock Test (verifies pipeline, schema, and Parquet writing with synthetic records):
  python3 build_merged_dataset.py --mode mock --limit 10

  # 2. Mock Test with Upload to KhangTruong/Merged (cross_test split):
  python3 build_merged_dataset.py --mode mock --limit 10 --upload

  # 3. Live Scout/Stream Run with SAGI-D (via Kaggle API) + CocoGlide:
  python3 build_merged_dataset.py --mode live --limit 5 --datasets sagid cocoglide --upload

  # 4. Full Production Run (1,000 real and 1,000 fake per dataset):
  python3 build_merged_dataset.py --mode live --limit 1000 --upload
"""

import argparse
import sys
from typing import List

from src.builder import DatasetBuilder
from src.sources import (
    BaseDatasetSource,
    BRGenSource,
    CocoGlideSource,
    MockDatasetSource,
    OpenSDIDSource,
    SAGIDSource,
    TGIFSource,
)
from src.uploader import HubUploader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build and upload multi-generator AI tampering dataset to KhangTruong/Merged"
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
        default="KhangTruong/Merged",
        help="Target Hugging Face dataset repository",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Whether to upload the generated shards and README to Hugging Face",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dist_data",
        help="Directory to save generated Parquet files",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional list of specific datasets to include (e.g. sagid cocoglide opensdid_flux opensdid_sd15 tgif brgen)",
    )
    parser.add_argument(
        "--brgen-dir",
        type=str,
        default=None,
        help="Optional local path to unpacked BR-Gen directory (containing Au/, Tp/, Gt/)",
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
        MockDatasetSource("BR-Gen", method="PowerPaint"),
    ]


def get_live_sources(filter_names: List[str] = None, brgen_dir: str = None) -> List[BaseDatasetSource]:
    all_sources = [
        SAGIDSource(),
        CocoGlideSource(),
        OpenSDIDSource(target_generator="Flux.1"),
        OpenSDIDSource(target_generator="SD1.5"),
        OpenSDIDSource(target_generator="SDXL"),
        TGIFSource(target_model="flux1dev"),
        BRGenSource(data_dir=brgen_dir),
    ]
    if not filter_names:
        return all_sources

    filter_lower = [f.lower() for f in filter_names]
    selected = [s for s in all_sources if any(k in s.name.lower() for k in filter_lower)]
    return selected if selected else all_sources


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("AI IMAGE TAMPERING DATASET MERGER & BUILDER")
    print("=" * 60)
    print(f"Mode:         {args.mode}")
    print(f"Target split: {args.split}")
    print(f"Target repo:  {args.repo}")
    print(f"Limit/source: {args.limit} real, {args.limit} fake")
    print(f"Upload:       {args.upload}")
    print(f"Output dir:   {args.output_dir}")
    print("=" * 60 + "\n")

    # 1. Setup sources
    if args.mode == "mock":
        sources = get_mock_sources(args.limit)
    else:
        sources = get_live_sources(args.datasets, brgen_dir=args.brgen_dir)

    print(f"Active sources ({len(sources)}):")
    for s in sources:
        print(f"  - {s.name}")

    # 2. Build dataset
    builder = DatasetBuilder(output_dir=args.output_dir)
    shard_paths = builder.build_split(
        sources=sources,
        split_name=args.split,
        target_real_per_dataset=args.limit,
        target_fake_per_dataset=args.limit,
        batch_size=500,
    )

    if not shard_paths:
        print("Error: No shards generated. Exiting.")
        sys.exit(1)

    # 3. Collect summary statistics
    datasets_included = [s.name for s in sources]
    methods_included = []
    for s in sources:
        if hasattr(s, "_method"):
            methods_included.append(s._method)
        elif hasattr(s, "target_generator"):
            methods_included.append(s.target_generator)
        elif hasattr(s, "target_model") and s.target_model:
            methods_included.append(s.target_model)
        elif "CocoGlide" in s.name:
            methods_included.append("GLIDE")
        elif "SAGID" in s.name:
            methods_included.extend(["brushnet", "powerpaint", "controlnet", "hdpainter"])
        elif "BR-Gen" in s.name:
            methods_included.extend(["PowerPaint", "BrushNet", "SDXL", "LaMa", "MAT"])
    methods_included.append("authentic")

    # 4. Upload if requested
    if args.upload:
        # Read total rows from generated shards
        import pyarrow.parquet as pq

        total_samples = sum(len(pq.read_table(p)) for p in shard_paths)
        uploader = HubUploader(repo_id=args.repo)
        uploader.upload_shards(
            shard_paths=shard_paths,
            split_name=args.split,
            total_samples=total_samples,
            datasets_included=datasets_included,
            methods_included=methods_included,
        )
    else:
        print("\nSkipping upload (use --upload to push to Hugging Face).")
        print(f"Generated shards available locally in: {args.output_dir}/")


if __name__ == "__main__":
    main()
