#!/usr/bin/env python3
"""
CLI script to download, unpack, and verify image tampering and manipulation localization benchmarks.

Usage Examples:
  # 1. Download the standard benchmark bundle (Columbia, Coverage, CASIA1, CASIA2, NIST16, IMD2020):
  python3 download_datasets.py --datasets standard_bundle

  # 2. Download individual datasets:
  python3 download_datasets.py --datasets casia1 coverage columbia

  # 3. Download with custom target directory and disk limit:
  python3 download_datasets.py --datasets cocoglide imd2020_inpainting --target-dir /path/to/data --max-disk-gb 100

  # 4. List all downloadable datasets:
  python3 download_datasets.py --list
"""

import argparse
import os
import sys

from src.downloader import DATASET_DOWNLOAD_CATALOG, check_disk_safety, download_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and prepare AI image tampering datasets with disk safety"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["standard_bundle"],
        help=f"Datasets to download. Options: {', '.join(DATASET_DOWNLOAD_CATALOG.keys())} or 'all'",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="data",
        help="Directory to save downloaded and extracted dataset files (default: data/)",
    )
    parser.add_argument(
        "--max-disk-gb",
        type=float,
        default=100.0,
        help="Maximum disk space limit in GB before halting (default: 100.0)",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Retain raw zip/tar archives after extraction (default: False, deleted right away to save disk)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all supported downloadable datasets and exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("\n" + "=" * 70)
        print("SUPPORTED IMAGE TAMPERING BENCHMARKS & DOWNLOAD SOURCES")
        print("=" * 70)
        for name, info in DATASET_DOWNLOAD_CATALOG.items():
            gb = info.get("estimated_gb", "varies")
            print(f"- {name:<20} [{info['provider']:<14}] (~{gb} GB): {info['description']}")
        print("=" * 70 + "\n")
        return

    requested = args.datasets
    if "all" in requested:
        requested = list(DATASET_DOWNLOAD_CATALOG.keys())

    print("\n" + "=" * 70)
    print("IMAGE TAMPERING DATASET DOWNLOADER")
    print("=" * 70)
    print(f"Target directory:    {args.target_dir}")
    print(f"Max disk ceiling:    {args.max_disk_gb} GB")
    print(f"Delete archives:     {not args.keep_archives}")
    print(f"Datasets requested:  {', '.join(requested)}")
    print("=" * 70 + "\n")

    os.makedirs(args.target_dir, exist_ok=True)
    check_disk_safety(args.target_dir, estimated_gb=2.0, max_limit_gb=args.max_disk_gb)

    successes = []
    failures = []

    for name in requested:
        try:
            path = download_dataset(
                dataset_name=name,
                target_dir=args.target_dir,
                max_disk_gb=args.max_disk_gb,
                delete_archive=not args.keep_archives,
            )
            successes.append((name, path))
        except Exception as e:
            print(f"\n[Error] Failed to download '{name}': {e}\n")
            failures.append((name, str(e)))

    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    if successes:
        print(f"Successfully prepared {len(successes)} dataset(s):")
        for name, path in successes:
            print(f"  ✓ {name:<18} -> {path}")
    if failures:
        print(f"\nFailed to prepare {len(failures)} dataset(s):")
        for name, err in failures:
            print(f"  ✗ {name:<18} -> {err}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
