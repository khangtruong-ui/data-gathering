"""
Hugging Face Hub uploader: commits Parquet shards and updates the dataset README card.
"""

import os
from typing import List, Optional
from huggingface_hub import HfApi


def generate_dataset_card(
    repo_id: str,
    split_name: str,
    total_samples: int,
    datasets_included: List[str],
    methods_included: List[str],
) -> str:
    """Generates the README.md dataset card with Hugging Face YAML metadata."""
    datasets_str = "\n".join([f"- **{d}**" for d in sorted(list(set(datasets_included)))])
    methods_str = "\n".join([f"- `{m}`" for m in sorted(list(set(methods_included)))])

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
- `method` (`string`): Specific tampering method or generator used (e.g., `Flux.1`, `SD1.5`, `SDXL`, `GLIDE`, `authentic`).
- `mask_ratio` (`float32`): Percentage of pixels tampered (`0.0` for real, `0.0 < ratio < 0.95` for localized edits).

## Datasets Included
{datasets_str}

## Generators & Methods
{methods_str}

## Accessibility & Data Source Notes
- **OpenSDID:** Sourced from arXiv:2503.19653. Filtered for strictly localized inpainting/tampering (excluding 100% fake full-generation images).
- **CocoGlide:** Sourced from TruFor CVPR 2023. Contains GLIDE inpainting on MS-COCO backgrounds.
- **TGIF & TGIF2:** Sourced from WIFS 2024 / JIS 2026. Only the spliced (`-sp`) subset is included to ensure authentic background retention.
- **BR-Gen:** AAAI 2026. Distributed via multi-part zips (`.z01`...`.z12`); requires local archive extraction.
- **SAGI-D:** arXiv:2502.06593. Available via Kaggle API (`giakop/sagi-d`); requires Kaggle credentials.
- **Inpaint32K & AutoSplice:** Baidu Netdisk / Google Form gated.
"""
    return card


class HubUploader:
    def __init__(self, repo_id: str = "KhangTruong/Merged"):
        self.repo_id = repo_id
        self.api = HfApi()

    def upload_shards(
        self,
        shard_paths: List[str],
        split_name: str,
        total_samples: int,
        datasets_included: List[str],
        methods_included: List[str],
    ):
        print(f"\n=======================================================")
        print(f"Uploading to Hugging Face: {self.repo_id}")
        print(f"=======================================================\n")

        # 1. Upload parquet shards into data/
        from tqdm import tqdm

        for p in tqdm(shard_paths, desc="Uploading shards to Hugging Face", unit="shard", dynamic_ncols=True):
            fname = os.path.basename(p)
            remote_path = f"data/{fname}"
            self.api.upload_file(
                path_or_fileobj=p,
                path_in_repo=remote_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"Add {split_name} shard {fname}",
            )

        # 2. Upload README.md dataset card
        readme_content = generate_dataset_card(
            repo_id=self.repo_id,
            split_name=split_name,
            total_samples=total_samples,
            datasets_included=datasets_included,
            methods_included=methods_included,
        )

        temp_readme = "/tmp/MERGED_README.md"
        with open(temp_readme, "w", encoding="utf-8") as f:
            f.write(readme_content)

        print("Uploading updated README.md dataset card...")
        self.api.upload_file(
            path_or_fileobj=temp_readme,
            path_in_repo="README.md",
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"Update dataset card for {split_name} split",
        )
        print(f"\nSuccessfully uploaded to https://huggingface.co/datasets/{self.repo_id}!")
