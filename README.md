# AI Image Tampering & Manipulation Localization Benchmark Pipeline

This repository gathers, filters, and standardizes localized AI image tampering datasets into a unified benchmark, uploading to Hugging Face under [`KhangTruong/Merged`](https://huggingface.co/datasets/KhangTruong/Merged) (split: `cross_test`).

## Requirements & Design Decisions

### 1. Schema Specifications
Each sample contains:
- `id` (`string`): Unique identifier for the sample (e.g., `opensdid_flux_fake_0001`, `sagid_coco_000000000110_brushnet_epicrealism_1_Blended`, `cocoglide_139871`).
- `image` (`image`): Image bytes (RGB).
- `mask` (`image`): Single-channel ground-truth mask:
  - **Real images:** Pure **black mask** (all zeros), synthesized with the exact same width and height as the image.
  - **Fake images:** Pixel-level binary mask where non-zero pixels (255) denote the tampered/inpainted regions.
- `label` (`int`): `0` for real (authentic), `1` for fake (manipulated).
- `label_text` (`string`): `"real"` or `"fake"`.
- `dataset` (`string`): Name of the originating benchmark (e.g. `OpenSDID_Flux`, `SAGI-D`, `CocoGlide`, `TGIF2_SD2`, `BR-Gen`).
- `method` (`string`): Specific tampering generator or technique (e.g., `Flux.1`, `SD1.5`, `SDXL`, `SD3`, `GLIDE`, `brushnet_epicrealism`, `PowerPaint`, `authentic`).
- `mask_ratio` (`float`): Percentage of pixels modified by inpainting/tampering (`0.0` for real, `0.0005 <= ratio <= 0.95` for localized tampering).

### 2. Avoiding 100% Fake (Fully Synthesized) Images
To strictly isolate **localized image tampering** and prevent whole-frame generations from polluting the localization benchmark:
- **SAGI-D:** Filtered for spliced inpaintings (`type == 'sp'`) where only the masked region was edited into an authentic photo. Fully regenerated variants (`type == 'fr'`) are strictly excluded.
- **OpenSDID:** Filter out samples where `mask is None` or where `mask_ratio >= 0.95`. Only rows with active, localized inpainting masks are retained.
- **TGIF & TGIF2:** Only the **spliced (`-sp`)** variants are included. The fully regenerated (`-fr`) variants are discarded.
- **CocoGlide:** All 512 images are localized inpaintings over authentic MS-COCO backgrounds.
- **BR-Gen:** Ground-truth masks in `Gt/` are verified to ensure `0.0005 <= mask_ratio <= 0.95`.

### 3. Disk-Safe Streaming
To prevent large archive downloads from exhausting local disk space:
- **SAGI-D:** Files are downloaded on-demand per sample via the Kaggle API and deleted immediately after packaging into the Parquet table.
- **OpenSDID:** Parquet shards are streamed from Hugging Face (`nebula/OpenSDI_test`) and processed row-by-row.
- **CocoGlide:** Streamed using HTTP Range requests directly from Grip Unina without unzipping the entire archive to disk.
- **TGIF2:** Spliced metadata and corresponding images are streamed on-demand.

---

## Dataset Accessibility & Status Matrix

| Dataset | Status | Access Method | Notes / Automation Details |
| :--- | :--- | :--- | :--- |
| **SAGI-D (DiQuID)** | ✅ **Fully Automated** | Kaggle API (`giakop/sagi-d`) | Uses `~/.kaggle/kaggle.json`. Downloads metadata and per-file assets on demand. |
| **OpenSDID** | ✅ **Direct Stream** | Hugging Face Hub (`nebula/OpenSDI_test`) | Public Parquet shards; generator partitioned. |
| **CocoGlide** | ✅ **Direct Stream** | HTTP Range / Zip (`grip.unina.it`) | 512 GLIDE inpainting pairs; no login required. |
| **TGIF & TGIF2** | ✅ **Direct Stream** | Nextcloud direct links + GitHub metadata | Filtered to `-sp` (spliced) only. |
| **BR-Gen** | ⚡ **Automated Loader** | Local directory / `--brgen-dir` | Parsed via `Au/`, `Tp/`, and `Gt/` folder hierarchy. |
| **Inpaint32K** | ⚠️ **Region Gated** | Baidu Netdisk | Requires Baidu Pan account/client; not directly streamable over standard HTTP. |
| **AutoSplice** | ⚠️ **Manual Form** | Google Drive (Request Form) | Requires manual form approval before download access. |

---

## Quickstart & Usage

### 1. Mock Test (Zero Bandwidth Dry Run)
Runs synthetic mock generators across all dataset sources (including SAGI-D and BR-Gen) to verify schema, Parquet serialization, and data integrity:
```bash
python3 build_merged_dataset.py --mode mock --limit 10
```

### 2. Live Run with SAGI-D (Kaggle API) & CocoGlide
Streams real inpainting pairs directly and generates Parquet shards:
```bash
python3 build_merged_dataset.py --mode live --datasets sagid cocoglide --limit 10 --upload
```

### 3. Live Run with BR-Gen
If you have downloaded or extracted BR-Gen:
```bash
python3 build_merged_dataset.py --mode live --datasets brgen sagid cocoglide --brgen-dir /path/to/BR-Gen --limit 1000 --upload
```

### 4. Full Multi-Generator Production Run (1,000 Real & 1,000 Fake per Dataset)
Streams up to 1,000 real and 1,000 fake images per source without disk blowup:
```bash
python3 build_merged_dataset.py --mode live --limit 1000 --upload
```

---

## Remote Dataset Hub
The dataset is hosted at:
**[https://huggingface.co/datasets/KhangTruong/Merged](https://huggingface.co/datasets/KhangTruong/Merged)**
