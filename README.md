# AI Image Tampering & Manipulation Localization Benchmark Pipeline

This repository gathers, filters, and standardizes localized AI image tampering datasets into a unified benchmark, uploading to Hugging Face under [`KhangTruong/Merged`](https://huggingface.co/datasets/KhangTruong/Merged) (split: `cross_test`).

## Requirements & Design Decisions

### 1. Schema Specifications
Each sample contains:
- `id` (`string`): Unique identifier for the sample (e.g., `opensdid_flux_fake_0001`, `cocoglide_139871`).
- `image` (`image`): Image bytes (RGB).
- `mask` (`image`): Single-channel ground-truth mask:
  - **Real images:** Pure **black mask** (all zeros), synthesized with the exact same width and height as the image.
  - **Fake images:** Pixel-level binary mask where non-zero pixels (255) denote the tampered/inpainted regions.
- `label` (`int`): `0` for real (authentic), `1` for fake (manipulated).
- `label_text` (`string`): `"real"` or `"fake"`.
- `dataset` (`string`): Name of the originating benchmark (e.g. `OpenSDID_Flux`, `OpenSDID_SD15`, `CocoGlide`, `TGIF2_SD2`).
- `method` (`string`): Specific tampering generator or technique (e.g., `Flux.1`, `SD1.5`, `SDXL`, `SD3`, `GLIDE`, `SD2_spliced`, `authentic`).
- `mask_ratio` (`float`): Percentage of pixels modified by inpainting/tampering (`0.0` for real, `0.0005 <= ratio <= 0.95` for localized tampering).

### 2. Avoiding 100% Fake (Fully Synthesized) Images
To strictly isolate **localized image tampering** and prevent whole-frame generations from polluting the localization benchmark:
- **OpenSDID:** Filter out samples where `mask is None` or where `mask_ratio >= 0.95`. Only rows with active, localized inpainting masks are retained.
- **TGIF & TGIF2:** Only the **spliced (`-sp`)** variants are included (where only the masked region is modified and blended back into the authentic photo). The fully regenerated (`-fr`) variants (which synthesize the entire image from scratch) are strictly excluded.
- **CocoGlide:** All 512 images are localized inpaintings over MS-COCO authentic backgrounds.

### 3. Disk-Safe Streaming
To prevent large archive downloads from exhausting local disk space:
- **OpenSDID:** Parquet shards are streamed from Hugging Face (`nebula/OpenSDI_test`) and processed row-by-row.
- **CocoGlide:** Streamed using HTTP Range requests directly from Grip Unina without unzipping the entire archive to disk.
- **TGIF2:** Spliced metadata and corresponding high-resolution source images are streamed on-demand.

---

## Dataset Accessibility & Status Matrix

| Dataset | Status | Access Method | Notes / Potential Bottlenecks |
| :--- | :--- | :--- | :--- |
| **OpenSDID** | ✅ **Direct Stream** | Hugging Face Hub (`nebula/OpenSDI_test`) | Public Parquet shards; generator partitioned. |
| **CocoGlide** | ✅ **Direct Stream** | HTTP Range / Zip (`grip.unina.it`) | 512 GLIDE inpainting pairs; no login required. |
| **TGIF & TGIF2** | ✅ **Direct Stream** | Nextcloud direct links + GitHub metadata | Filtered to `-sp` (spliced) only. |
| **BR-Gen** | ⚠️ **Local Archive** | Hugging Face (`clpbc/BR-Gen`) / Google Drive | Distributed as 13-part multi-part split zips (`.z01`...`.z12`); requires ~50GB disk to assemble. |
| **SAGI-D (DiQuID)** | ⚠️ **API Gated** | Kaggle (`giakop/sagi-d`) | Requires local `~/.kaggle/kaggle.json` credentials for automated CLI downloads. |
| **Inpaint32K** | ⚠️ **Region Gated** | Baidu Netdisk | Requires Baidu Pan account/client; not directly streamable over standard HTTP. |
| **AutoSplice** | ⚠️ **Manual Form** | Google Drive (Request Form) | Requires manual form approval before download access. |

---

## Quickstart & Usage

### Setup
```bash
pip install -r requirements.txt
```

### 1. Mock Test (Local Dry Run)
Runs synthetic mock generators to verify schema, Parquet serialization, and data integrity:
```bash
python3 build_merged_dataset.py --mode mock --limit 10
```

### 2. Mock Test with Upload
Generates mock test shards and uploads them to `KhangTruong/Merged` under split `cross_test`:
```bash
python3 build_merged_dataset.py --mode mock --limit 5 --upload
```

### 3. Live Streaming Run (Small Test)
Streams 10 real and 10 fake samples from real benchmarks (OpenSDID, CocoGlide, TGIF) and uploads:
```bash
python3 build_merged_dataset.py --mode live --limit 10 --upload
```

### 4. Full Production Run (1,000 Real & 1,000 Fake per Dataset)
Streams up to 1,000 real and 1,000 fake images per source without disk blowup:
```bash
python3 build_merged_dataset.py --mode live --limit 1000 --upload
```

---

## Remote Dataset Hub
The dataset is hosted at:
**[https://huggingface.co/datasets/KhangTruong/Merged](https://huggingface.co/datasets/KhangTruong/Merged)**
