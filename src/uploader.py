"""
Hugging Face Hub uploader: commits Parquet shards and dynamically updates the dataset README card.

Features:
- Checkpoint-aware uploading: skips already-uploaded shards and supports resuming.
- Dynamic metadata calculation: updates 'num_examples', included datasets, and methods breakdown.
- Automatically includes the researched Dataset Accessibility & Status Matrix with 'data_source'.
- Accurately reconciles document-data mismatches (e.g. BR-Gen absence and stray shards).
- Safe disk management: supports deleting local shards immediately after upload.
"""

import collections
import os
import tempfile
from typing import Dict, List, Optional, Union
from huggingface_hub import HfApi
import pyarrow.parquet as pq


# Standard investigated image tampering and manipulation localization benchmarks
INVESTIGATED_DATASETS_MATRIX = [
    {
        "dataset": "CASIA V1.0",
        "data_source": "https://github.com/namtpham/casia1groundtruth",
        "status": "✅ Available (Mirror)",
        "notes": "Original CASIA server is offline. Standardized masks and image pairs hosted on GitHub; also mirrored in HiFi-IFDL data bundle.",
    },
    {
        "dataset": "CASIA V2.0",
        "data_source": "https://github.com/namtpham/casia2groundtruth",
        "status": "✅ Available (Mirror)",
        "notes": "Original server offline. Provided via GitHub & Kaggle (divg07/casia-20-image-tampering-detection-dataset) with renaming/mask corrections.",
    },
    {
        "dataset": "Columbia Gray",
        "data_source": "https://www.ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/AuthSplicedDataSet.htm",
        "status": "⚠️ Form-Gated / Mirror",
        "notes": "Columbia DVMM lab requires form agreement. 128x128 uncompressed grayscale BMP blocks; bundled in HiFi-IFDL benchmark package.",
    },
    {
        "dataset": "NIST16 (Nimble 2016 / NC2016)",
        "data_source": "https://data.nist.gov/od/id/mds2-2410",
        "status": "⚠️ Registration-Gated",
        "notes": "Hosted on NIST OpenMFC portal (mfc.nist.gov); requires account registration and Data Transfer Agreement. Deduplication tools on GitHub.",
    },
    {
        "dataset": "Coverage",
        "data_source": "https://github.com/wenbihan/coverage",
        "status": "✅ Fully Accessible",
        "notes": "Official repository for COVERAGE copy-move benchmark (100 image pairs). Ground truth masks provided in MAT/TIFF format.",
    },
    {
        "dataset": "IMD2020_Inpainting",
        "data_source": "https://huggingface.co/datasets/KhangTruong/IMD2020",
        "status": "✅ Fully Accessible (HF Hub)",
        "notes": "35,000 synthetic inpainting images with localized binary masks. Streamable directly via Hugging Face Hub Parquet shards.",
    },
    {
        "dataset": "IMD2020_RealLife",
        "data_source": "http://staff.utia.cas.cz/novozada/db/",
        "status": "✅ Accessible (Author Web / Bundle)",
        "notes": "2,010 real-world manipulated images collected from the internet with matching pristine sources and manual masks; also in HiFi-IFDL bundle.",
    },
    {
        "dataset": "TampCOCO",
        "data_source": "https://github.com/mjkwon2021/CAT-Net",
        "status": "✅ Accessible (Kaggle / Baidu)",
        "notes": "Introduced in CAT-Net for JPEG compression artifact tracing on MS-COCO. Available on Kaggle (cauchycomplete/cat-net-tampcoco-image-splicing-dataset).",
    },
    {
        "dataset": "COCO-Inpaint",
        "data_source": None,
        "status": "❌ Not Yet Released",
        "notes": "Benchmark introduced in arXiv:2504.18361 (Yan et al., April 2025). The authors stated the 238K dataset will be released, but no public download URL is active yet. (Alternative: KhangTruong/COCO-inpainted on HF).",
    },
    {
        "dataset": "COCO-Glide",
        "data_source": "https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip",
        "status": "✅ Fully Automated (Range Stream)",
        "notes": "TruFor CVPR 2023 benchmark. 512 GLIDE inpainting pairs on MS-COCO backgrounds. Directly streamable over HTTP without full download.",
    },
    {
        "dataset": "AutoSplice",
        "data_source": "https://github.com/shanface33/AutoSplice_Dataset",
        "status": "⚠️ Drive / Baidu Gated",
        "notes": "CVPRW 2023 DALL-E 2 prompt-manipulated dataset. Download links provided via Google Drive and Baidu Wangpan in GitHub README.",
    },
    {
        "dataset": "HiFi_IFDL",
        "data_source": "https://github.com/CHELSEA234/HiFi_IFDL",
        "status": "✅ Accessible (Google Drive)",
        "notes": "CVPR 2023 / IJCV 2024 hierarchical forgery benchmark. Dataset and bundled standard benchmarks (Columbia, Coverage, CASIA, NIST16, IMD2020) hosted on Google Drive.",
    },
    {
        "dataset": "OpenSDID (Flux.1 / SD1.5 / SDXL)",
        "data_source": "https://huggingface.co/datasets/nebula/OpenSDI_test",
        "status": "✅ Fully Automated (Streamed)",
        "notes": "arXiv:2503.19653. Streamed row-group by row-group from Hugging Face Hub; filtered for strictly localized inpaintings.",
    },
    {
        "dataset": "SAGI-D (DiQuID)",
        "data_source": "https://www.kaggle.com/datasets/giakop/sagi-d",
        "status": "✅ Automated via Kaggle API",
        "notes": "arXiv:2502.06593. Streamed per-sample using Kaggle API credentials; filtered for spliced (type == 'sp') inpaintings.",
    },
    {
        "dataset": "TGIF & TGIF2",
        "data_source": "https://github.com/IDLabMedia/tgif-dataset",
        "status": "✅ Fully Automated (Streamed)",
        "notes": "WIFS 2024 / JIS 2026. Metadata on GitHub + MS-COCO image retrieval; restricted to spliced (-sp) subsets.",
    },
    {
        "dataset": "BR-Gen",
        "data_source": "https://huggingface.co/datasets/clpbc/BR-Gen",
        "status": "⚡ Requires Local Extraction",
        "notes": "AAAI 2026. Distributed as multi-part split archives (.z01..z12). Requires manual unpacking into local directory (--brgen-dir). Note: yields 0 samples if archive is not extracted locally.",
    },
]


def generate_dataset_card(
    repo_id: str,
    split_name: str,
    total_samples: int,
    datasets_distribution: Union[List[str], Dict[str, int]],
    methods_distribution: Union[List[str], Dict[str, int]],
    matrix: Optional[List[dict]] = None,
) -> str:
    """
    Generates an accurate, dynamic dataset README card with Hugging Face YAML metadata.
    Filters out any sources with 0 samples to ensure complete document-data consistency.
    """
    if matrix is None:
        matrix = INVESTIGATED_DATASETS_MATRIX

    # Build datasets list/breakdown
    if isinstance(datasets_distribution, dict):
        active_datasets = [
            f"- **{d}**: {count:,} samples"
            for d, count in sorted(datasets_distribution.items())
            if count > 0
        ]
    else:
        active_datasets = [f"- **{d}**" for d in sorted(list(set(datasets_distribution)))]

    # Build methods list/breakdown
    if isinstance(methods_distribution, dict):
        active_methods = [
            f"- `{m}`: {count:,} samples"
            for m, count in sorted(methods_distribution.items())
            if count > 0
        ]
    else:
        active_methods = [f"- `{m}`" for m in sorted(list(set(methods_distribution)))]

    datasets_str = "\n".join(active_datasets) if active_datasets else "- None"
    methods_str = "\n".join(active_methods) if active_methods else "- None"

    # Build markdown status matrix with data_source column
    matrix_rows = []
    for item in matrix:
        d_name = item["dataset"]
        src = item["data_source"]
        src_link = f"[{src}]({src})" if src else "None"
        status = item["status"]
        notes = item["notes"]
        matrix_rows.append(f"| **{d_name}** | {src_link} | {status} | {notes} |")

    matrix_str = "\n".join(matrix_rows)

    card = f"""---
license: cc-by-sa-4.0
task_categories:
- image-segmentation
- image-classification
tags:
- image-tampering
- inpainting
- deepfake-localization
- generative-forgery
- diffusion
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: {split_name}
    path: data/{split_name}-*
dataset_info:
  features:
  - name: id
    dtype: string
  - name: image
    dtype: image
  - name: mask
    dtype: image
  - name: label
    dtype:
      class_label:
        names:
          '0': real
          '1': fake
  - name: label_text
    dtype: string
  - name: dataset
    dtype: string
  - name: method
    dtype: string
  - name: mask_ratio
    dtype: float32
  splits:
  - name: {split_name}
    num_examples: {total_samples}
---

# {repo_id} ({split_name})

This benchmark combines multi-generator AI image tampering and manipulation localization datasets.
Every sample contains an image and a ground-truth binary mask indicating tampered regions.

## Columns
- `id` (`string`): Unique identifier for the sample.
- `image` (`image`): The image (authentic or manipulated).
- `mask` (`image`): Single-channel ground-truth mask. For **real images**, this is an **all-black mask** (0). For **fake images**, white pixels (255) denote the tampered/inpainted regions.
- `label` (`ClassLabel`): `0` = real, `1` = fake.
- `label_text` (`string`): `"real"` or `"fake"`.
- `dataset` (`string`): Originating benchmark / dataset name.
- `method` (`string`): Specific tampering method or generator used.
- `mask_ratio` (`float32`): Percentage of pixels tampered (`0.0` for real, `0.0005 <= ratio <= 0.95` for localized edits).

## Datasets Actually Present in this Split ({total_samples:,} Total Samples)
{datasets_str}

## Generators & Methods Present
{methods_str}

---

## Dataset Accessibility & Source Matrix

| Dataset | data_source | Status | Notes / Difficulties / Access Details |
| :--- | :--- | :--- | :--- |
{matrix_str}

---

## Document-Data Reconciliation Notes
- **BR-Gen Investigation:** BR-Gen is distributed as split archives (`.z01`...`.z12` on HF / Baidu Netdisk). In previous builds where local extraction was omitted, BR-Gen yielded 0 records. The automated dataset card generator now verifies actual ingested counts, preventing 0-row datasets from being erroneously advertised as present.
- **COCO-Inpaint Status:** The official benchmark from arXiv:2504.18361 (Yan et al.) has not yet activated a public download repository; marked as `None` in `data_source`.
- **Disk & Memory Safety:** Ingestion pipelines utilize bounded streaming queues and incremental Parquet sharding, avoiding in-memory accumulation and enforcing a strict 100GB disk ceiling with immediate cleanup.
"""
    return card


class HubUploader:
    def __init__(self, repo_id: str = "KhangTruong/Tampered-Images-Combined-Dataset-2"):
        self.repo_id = repo_id
        self.api = HfApi()

    def get_remote_repo_files(self) -> List[str]:
        try:
            return self.api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")
        except Exception as e:
            print(f"Warning: Could not list remote repo files ({e})")
            return []

    def upload_single_shard(self, shard_path: str, split_name: str) -> bool:
        """Uploads a single shard file to the Hub."""
        fname = os.path.basename(shard_path)
        remote_path = f"data/{fname}"
        try:
            self.api.upload_file(
                path_or_fileobj=shard_path,
                path_in_repo=remote_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"Add {split_name} shard {fname}",
            )
            return True
        except Exception as e:
            print(f"Error uploading shard {fname}: {e}")
            return False

    def update_dataset_card(
        self,
        split_name: str,
        total_samples: int,
        datasets_distribution: Union[List[str], Dict[str, int]],
        methods_distribution: Union[List[str], Dict[str, int]],
    ):
        """Generates and uploads the updated README.md dataset card."""
        readme_content = generate_dataset_card(
            repo_id=self.repo_id,
            split_name=split_name,
            total_samples=total_samples,
            datasets_distribution=datasets_distribution,
            methods_distribution=methods_distribution,
        )

        temp_readme = os.path.join(tempfile.gettempdir(), "MERGED_README.md")
        with open(temp_readme, "w", encoding="utf-8") as f:
            f.write(readme_content)

        print(f"Uploading updated README.md dataset card (num_examples: {total_samples:,})...")
        self.api.upload_file(
            path_or_fileobj=temp_readme,
            path_in_repo="README.md",
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"Update dataset card: {split_name} ({total_samples} examples)",
        )
        try:
            os.remove(temp_readme)
        except OSError:
            pass

    def upload_shards(
        self,
        shard_paths: List[str],
        split_name: str,
        total_samples: int,
        datasets_included: Union[List[str], Dict[str, int]],
        methods_included: Union[List[str], Dict[str, int]],
        delete_after_upload: bool = False,
    ):
        print(f"\n=======================================================")
        print(f"Uploading to Hugging Face: {self.repo_id}")
        print(f"Total shards: {len(shard_paths)}, Total samples: {total_samples:,}")
        print(f"=======================================================\n")

        remote_files = set(self.get_remote_repo_files())
        from tqdm import tqdm

        for p in tqdm(shard_paths, desc="Uploading Parquet shards", unit="shard", dynamic_ncols=True):
            if not os.path.exists(p):
                continue
            fname = os.path.basename(p)
            remote_path = f"data/{fname}"
            if remote_path in remote_files:
                print(f"  Shard {fname} already present on remote Hub. Skipping.")
                continue

            success = self.upload_single_shard(p, split_name)
            if success and delete_after_upload:
                try:
                    os.remove(p)
                    print(f"  [Disk-Safe] Deleted local shard {fname} after upload.")
                except Exception as e:
                    print(f"  Warning: could not delete {fname}: {e}")

        # Update dataset card with current accurate counts
        self.update_dataset_card(
            split_name=split_name,
            total_samples=total_samples,
            datasets_distribution=datasets_included,
            methods_distribution=methods_included,
        )
        print(f"\nSuccessfully updated dataset at https://huggingface.co/datasets/{self.repo_id}!")

    def _cleanup_legacy_shards(self, split_name: str, shards: List[str]) -> List[str]:
        """
        Cleans up legacy '-of-' shards if standard indexed shards ('{split_name}-{idx:05d}.parquet') exist,
        or reconciles mismatched '-of-' shard sets.
        """
        from huggingface_hub import CommitOperationDelete

        indexed_shards = [s for s in shards if "-of-" not in s]
        multi_shards = [s for s in shards if "-of-" in s]

        to_delete = []
        if indexed_shards and multi_shards:
            # Standard DatasetBuilder shards are present; any '-of-' shards are legacy orphans
            to_delete = multi_shards
            print(
                f"[Sync] Found {len(indexed_shards)} active indexed shards and "
                f"{len(multi_shards)} legacy '-of-' shards. Removing legacy shards..."
            )
        elif multi_shards:
            denoms = collections.Counter()
            for s in multi_shards:
                parts = s.split("-of-")
                if len(parts) == 2:
                    denom = parts[1].split(".")[0]
                    denoms[denom] += 1
            target_denom = denoms.most_common(1)[0][0]
            to_delete = [s for s in shards if target_denom and f"-of-{target_denom}.parquet" not in s]

        if to_delete:
            try:
                ops = [CommitOperationDelete(path_in_repo=s) for s in to_delete]
                self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    operations=ops,
                    commit_message=f"Remove {len(to_delete)} orphaned/legacy Parquet shards for {split_name}",
                )
                print(f"[Sync] Deleted {len(to_delete)} orphaned shards from {self.repo_id}.")
            except Exception as e:
                print(f"[Sync] Warning: batch delete failed ({e}), attempting individual deletes...")
                for s in to_delete:
                    try:
                        self.api.delete_file(
                            path_in_repo=s,
                            repo_id=self.repo_id,
                            repo_type="dataset",
                            commit_message=f"Remove orphaned shard {s}",
                        )
                    except Exception as inner_e:
                        print(f"[Sync] Could not delete {s}: {inner_e}")

            files = self.get_remote_repo_files()
            shards = sorted([f for f in files if f.startswith(f"data/{split_name}-") and f.endswith(".parquet")])

        return shards

    def reconstruct_checkpoint_from_hub(
        self,
        split_name: str = "cross_test",
        output_dir: str = "dist_data",
        target_real: int = 2000,
        target_fake: int = 2000,
        remove_orphaned_shards: bool = True,
    ) -> Optional[dict]:
        """
        Inspects existing Parquet shards on the remote Hugging Face repository and reconstructs
        a local checkpoint structure so '--resume' can seamlessly append new shards without
        re-downloading or duplicating datasets already uploaded to the Hub.
        """
        import fsspec

        files = self.get_remote_repo_files()
        shards = sorted([f for f in files if f.startswith(f"data/{split_name}-") and f.endswith(".parquet")])
        if not shards:
            return None

        if remove_orphaned_shards:
            shards = self._cleanup_legacy_shards(split_name, shards)

        indexed_shards = [s for s in shards if "-of-" not in s]
        if not indexed_shards:
            return None

        print(f"[Resume-Sync] Reconstructing checkpoint from {len(indexed_shards)} remote shards on {self.repo_id}...")
        fs = fsspec.filesystem("http")
        completed_shards = []
        source_progress: Dict[str, dict] = {}
        total_samples = 0

        for shard in indexed_shards:
            fname = os.path.basename(shard)
            url = f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{shard}"
            with fs.open(url, "rb") as f:
                pf = pq.ParquetFile(f)
                num_rows = pf.metadata.num_rows
                t = pf.read(columns=["dataset", "method", "label"])
                ds_list = t["dataset"].to_pylist()
                m_list = t["method"].to_pylist()
                lbl_list = t["label"].to_pylist()

            ds_counts = dict(collections.Counter(ds_list))
            m_counts = dict(collections.Counter(m_list))
            for d, l in zip(ds_list, lbl_list):
                norm_d = d.replace(".", "").replace("-", "")
                for key in {d, norm_d}:
                    if key not in source_progress:
                        source_progress[key] = {"real": 0, "fake": 0, "completed": True}
                    if l == 0:
                        source_progress[key]["real"] += 1
                    else:
                        source_progress[key]["fake"] += 1

            completed_shards.append(
                {
                    "filename": fname,
                    "path": os.path.join(output_dir, fname),
                    "num_rows": num_rows,
                    "datasets": ds_counts,
                    "methods": m_counts,
                    "uploaded": True,
                }
            )
            total_samples += num_rows

        return {
            "split_name": split_name,
            "target_real": target_real,
            "target_fake": target_fake,
            "completed_shards": completed_shards,
            "source_progress": source_progress,
            "total_samples": total_samples,
        }

    def sync_hf_repo_documentation(self, split_name: str = "cross_test", remove_orphaned_shards: bool = True):
        """
        Inspects the remote Hugging Face repository, cleans up any orphaned/mock shards,
        tallies true dataset distributions, and synchronizes the remote README card.
        """
        import fsspec

        print(f"\n[Sync] Inspecting remote repository {self.repo_id}...")
        files = self.get_remote_repo_files()

        # Find all cross_test parquet shards
        shards = sorted([f for f in files if f.startswith(f"data/{split_name}-") and f.endswith(".parquet")])
        print(f"[Sync] Found {len(shards)} Parquet shards for split '{split_name}'.")

        if remove_orphaned_shards and shards:
            shards = self._cleanup_legacy_shards(split_name, shards)

        fs = fsspec.filesystem("http")
        dataset_counter = collections.Counter()
        method_counter = collections.Counter()
        total_rows = 0

        print(f"[Sync] Reading metadata and row distributions across {len(shards)} shards...")
        for shard in shards:
            url = f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{shard}"
            with fs.open(url, "rb") as f:
                pf = pq.ParquetFile(f)
                total_rows += pf.metadata.num_rows
                t = pf.read(columns=["dataset", "method"])
                dataset_counter.update(t["dataset"].to_pylist())
                method_counter.update(t["method"].to_pylist())

        print(f"[Sync] Total verified rows on Hub: {total_rows:,}")
        print(f"[Sync] Active datasets on Hub: {dict(dataset_counter)}")

        self.update_dataset_card(
            split_name=split_name,
            total_samples=total_rows,
            datasets_distribution=dict(dataset_counter),
            methods_distribution=dict(method_counter),
        )
        print(f"[Sync] Documentation successfully synchronized for {self.repo_id}!")
