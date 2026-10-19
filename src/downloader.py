"""
Automated downloader for AI image tampering and manipulation localization benchmarks.

Features:
- Multi-provider support: Google Drive (gdown), Kaggle API, Hugging Face Hub, and HTTP.
- Strict disk safety enforcement: verifies available disk space and 100GB ceilings.
- Immediate archive deletion: unpacks tar/zip archives and deletes the raw archive immediately to prevent disk exhaustion.
"""

import os
import shutil
import tarfile
import tempfile
from typing import Dict, List, Optional
import zipfile

import requests
from tqdm import tqdm


DATASET_DOWNLOAD_CATALOG: Dict[str, dict] = {
    "standard_bundle": {
        "provider": "gdrive",
        "id": "1RYXTg0Q82KEvkeOtaaR5AZ0FBx5219SY",
        "archive_name": "data.tar.gz",
        "subfolder": "data",
        "description": "Standard 5-dataset benchmark bundle (Columbia, Coverage, CASIA1, CASIA2, NIST16, IMD2020)",
        "estimated_gb": 5.6,
    },
    "casia1": {
        "provider": "gdrive_bundle",
        "parent_bundle": "standard_bundle",
        "rel_path": "CASIA/CASIA1",
        "description": "CASIA V1.0 (800 authentic, 921 spliced images with binary masks)",
    },
    "casia2": {
        "provider": "kaggle",
        "dataset_ref": "divg07/casia-20-image-tampering-detection-dataset",
        "rel_path": "CASIA2",
        "description": "CASIA V2.0 (7,491 authentic, 5,123 tampered images with masks)",
        "estimated_gb": 2.2,
    },
    "columbia": {
        "provider": "gdrive_bundle",
        "parent_bundle": "standard_bundle",
        "rel_path": "columbia",
        "description": "Columbia Gray & Uncompressed Image Splicing Evaluation Dataset",
    },
    "coverage": {
        "provider": "gdrive_bundle",
        "parent_bundle": "standard_bundle",
        "rel_path": "Coverage",
        "description": "COVERAGE Copy-Move Forgery Dataset with Similar but Genuine Objects",
    },
    "nist16": {
        "provider": "gdrive_bundle",
        "parent_bundle": "standard_bundle",
        "rel_path": "NIST16",
        "description": "NIST16 (Nimble Challenge 2016 / OpenMFC Evaluation Dataset)",
    },
    "imd2020_reallife": {
        "provider": "gdrive_bundle",
        "parent_bundle": "standard_bundle",
        "rel_path": "IMD2020",
        "description": "IMD2020 Real-Life (2,010 real-world internet manipulated images with manual masks)",
    },
    "imd2020_inpainting": {
        "provider": "huggingface",
        "repo_id": "KhangTruong/IMD2020",
        "rel_path": "IMD2020_Inpainting",
        "description": "IMD2020 Inpainting (35,000 synthetic inpainting images/masks on HF Hub)",
        "estimated_gb": 12.0,
    },
    "cocoinpaint": {
        "provider": "huggingface",
        "repo_id": "KhangTruong/COCO-inpainted",
        "rel_path": "COCO-Inpaint",
        "description": "COCO-Inpaint benchmark (streamable Parquet shards on HF Hub)",
        "estimated_gb": 4.0,
    },
    "cocoglide": {
        "provider": "http_zip",
        "url": "https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip",
        "archive_name": "CocoGlide.zip",
        "rel_path": "CocoGlide",
        "description": "CocoGlide (512 GLIDE inpainting pairs on MS-COCO backgrounds)",
        "estimated_gb": 0.5,
    },
    "autosplice": {
        "provider": "gdrive_folder",
        "folder_id": "1QpBm4528ng877ytdBiRSnrH-rsrCeoIA",
        "rel_path": "AutoSplice",
        "description": "AutoSplice (CVPRW 2023 DALL-E 2 prompt-manipulated dataset)",
        "estimated_gb": 8.0,
    },
    "hifi_ifdl": {
        "provider": "gdrive_folder",
        "folder_id": "1fwBEmW30-e0ECpCNNG3nRU6I9OqJfMAn",
        "rel_path": "HiFi_IFDL",
        "description": "HiFi-IFDL (CVPR 2023 Hierarchical Fine-Grained Image Forgery Benchmark)",
        "estimated_gb": 10.0,
    },
    "sagid": {
        "provider": "kaggle",
        "dataset_ref": "giakop/sagi-d",
        "rel_path": "SAGI-D",
        "description": "SAGI-D DiQuID Inpainting Benchmark (via Kaggle API)",
        "estimated_gb": 15.0,
    },
}


def check_disk_safety(target_dir: str, estimated_gb: float = 5.0, max_limit_gb: float = 100.0):
    """
    Ensures adequate disk space exists and that the target directory does not exceed max_limit_gb.
    """
    os.makedirs(target_dir, exist_ok=True)
    usage = shutil.disk_usage(target_dir)
    free_gb = usage.free / (1024**3)

    if free_gb < (estimated_gb + 2.0):
        raise RuntimeError(
            f"INSUFFICIENT DISK SPACE: Need at least {estimated_gb + 2.0:.1f} GB free, "
            f"but only {free_gb:.1f} GB is available on filesystem."
        )

    # Check total size of target directory
    total_used = 0
    for root, _, files in os.walk(target_dir):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total_used += os.path.getsize(fp)
            except OSError:
                pass
    used_gb = total_used / (1024**3)

    if used_gb + estimated_gb > max_limit_gb:
        raise RuntimeError(
            f"DISK CEILING REACHED: Target directory ({used_gb:.1f} GB) + new download "
            f"({estimated_gb:.1f} GB) would exceed the maximum ceiling of {max_limit_gb:.1f} GB. "
            f"Please clean old datasets or increase --max-disk-gb."
        )


def extract_archive(archive_path: str, extract_to: str, delete_archive: bool = True) -> str:
    """
    Extracts tar/gz or zip archives and deletes the compressed archive immediately to save disk.
    """
    os.makedirs(extract_to, exist_ok=True)
    print(f"Extracting {archive_path} to {extract_to}...")

    if archive_path.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(path=extract_to)
    elif archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(path=extract_to)
    else:
        print(f"Notice: Unrecognized archive format {archive_path}; skipped extraction.")
        return extract_to

    print(f"Successfully extracted archive.")

    if delete_archive and os.path.exists(archive_path):
        try:
            os.remove(archive_path)
            print(f"[Disk-Safe] Deleted raw archive {os.path.basename(archive_path)} immediately after extraction.")
        except Exception as e:
            print(f"Warning: Could not remove archive {archive_path}: {e}")

    return extract_to


def download_http_file(url: str, output_path: str) -> str:
    """Downloads an HTTP/HTTPS file with a progress bar."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"Downloading from {url}...")
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with open(output_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc=os.path.basename(output_path)
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return output_path


def download_google_drive(file_id: str, output_path: str) -> str:
    """Downloads a public Google Drive file using gdown with automatic confirmation handling."""
    import gdown

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"Downloading Google Drive file ID: {file_id}...")
    output = gdown.download(id=file_id, output=output_path, quiet=False)
    if not output or not os.path.exists(output_path):
        raise RuntimeError(f"Failed to download Google Drive file {file_id}")
    return output_path


def download_google_drive_folder(folder_id: str, output_dir: str) -> str:
    """Downloads a public Google Drive folder using gdown."""
    import gdown

    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading Google Drive folder ID: {folder_id}...")
    gdown.download_folder(id=folder_id, output=output_dir, quiet=False)
    return output_dir


def download_kaggle_dataset(dataset_ref: str, target_dir: str) -> str:
    """Downloads and extracts a Kaggle dataset using ~/.kaggle/kaggle.json."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api._print_dataset_url_and_license = lambda *args, **kwargs: None

    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading Kaggle dataset {dataset_ref} into {target_dir}...")
    api.dataset_download_files(dataset_ref, path=target_dir, unzip=True, quiet=False)
    print(f"Kaggle download and unzipping complete.")
    return target_dir


def download_huggingface_dataset(repo_id: str, target_dir: str, allow_patterns: Optional[List[str]] = None) -> str:
    """Downloads Parquet files from Hugging Face Hub using huggingface_hub snapshot_download."""
    from huggingface_hub import snapshot_download

    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading Hugging Face dataset {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=target_dir,
        allow_patterns=allow_patterns or ["data/*.parquet", "*.parquet", "README.md"],
    )
    return target_dir


def download_dataset(
    dataset_name: str,
    target_dir: str = "data",
    max_disk_gb: float = 100.0,
    delete_archive: bool = True,
) -> str:
    """
    High-level download dispatcher for any supported dataset.
    """
    key = dataset_name.lower().replace("-", "_").replace(".", "").replace(" ", "_")
    if key not in DATASET_DOWNLOAD_CATALOG:
        # Check alias
        alias_map = {
            "casia_v1": "casia1",
            "casia_v10": "casia1",
            "casia10": "casia1",
            "casia_v2": "casia2",
            "casia_v20": "casia2",
            "casia20": "casia2",
            "columbia_gray": "columbia",
            "nist_16": "nist16",
            "imd2020": "imd2020_inpainting",
            "imd2020_real_life": "imd2020_reallife",
            "coco_inpaint": "cocoinpaint",
            "coco_glide": "cocoglide",
            "hifi": "hifi_ifdl",
        }
        key = alias_map.get(key, key)

    if key not in DATASET_DOWNLOAD_CATALOG:
        available = ", ".join(DATASET_DOWNLOAD_CATALOG.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")

    entry = DATASET_DOWNLOAD_CATALOG[key]
    provider = entry["provider"]
    est_gb = entry.get("estimated_gb", 5.0)

    check_disk_safety(target_dir, estimated_gb=est_gb, max_limit_gb=max_disk_gb)

    print(f"\n=======================================================")
    print(f"Downloading: {dataset_name} ({entry['description']})")
    print(f"Provider:    {provider}")
    print(f"Target dir:  {target_dir}")
    print(f"=======================================================\n")

    # 1. Standard bundle or nested bundle entry
    if provider == "gdrive_bundle":
        # Check if bundle is already unpacked in target_dir
        expected_path = os.path.join(target_dir, entry["rel_path"])
        if os.path.exists(expected_path) and len(os.listdir(expected_path)) > 0:
            print(f"Dataset already present at {expected_path}. Skipping download.")
            return expected_path

        # If standard bundle is not yet downloaded, download and unpack it
        bundle_entry = DATASET_DOWNLOAD_CATALOG["standard_bundle"]
        archive_path = os.path.join(target_dir, bundle_entry["archive_name"])
        download_google_drive(bundle_entry["id"], archive_path)
        extract_archive(archive_path, target_dir, delete_archive=delete_archive)
        return expected_path

    elif provider == "gdrive":
        archive_path = os.path.join(target_dir, entry.get("archive_name", "download.tar.gz"))
        download_google_drive(entry["id"], archive_path)
        dest = os.path.join(target_dir, entry.get("rel_path", ""))
        extract_archive(archive_path, dest if dest else target_dir, delete_archive=delete_archive)
        return dest or target_dir

    elif provider == "gdrive_folder":
        dest = os.path.join(target_dir, entry.get("rel_path", key))
        return download_google_drive_folder(entry["folder_id"], dest)

    elif provider == "kaggle":
        dest = os.path.join(target_dir, entry.get("rel_path", key))
        return download_kaggle_dataset(entry["dataset_ref"], dest)

    elif provider == "huggingface":
        dest = os.path.join(target_dir, entry.get("rel_path", key))
        return download_huggingface_dataset(entry["repo_id"], dest)

    elif provider == "http_zip":
        archive_path = os.path.join(target_dir, entry.get("archive_name", "archive.zip"))
        download_http_file(entry["url"], archive_path)
        dest = os.path.join(target_dir, entry.get("rel_path", key))
        extract_archive(archive_path, dest, delete_archive=delete_archive)
        return dest

    raise NotImplementedError(f"Provider {provider} not supported.")
