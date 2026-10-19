# AI Image Tampering & Manipulation Localization Benchmark Pipeline

This repository gathers, filters, standardizes, and uploads multi-generator localized AI image tampering datasets into a unified benchmark on Hugging Face under [`KhangTruong/Tampered-Images-Combined-Dataset-2`](https://huggingface.co/datasets/KhangTruong/Tampered-Images-Combined-Dataset-2) (split: `cross_test`).

---

## 1. Schema Specifications
Every sample in the merged dataset adheres to the following standardized schema:
- `id` (`string`): Unique identifier for the sample (e.g., `opensdid_flux_fake_0001`, `sagid_coco_000000000110_brushnet_epicrealism_1_Blended`, `cocoglide_139871`, `imd2020_robgsummers_5499245555`).
- `image` (`image`): Image bytes (RGB).
- `mask` (`image`): Single-channel ground-truth binary mask:
  - **Real images:** Pure **black mask** (all zeros), synthesized with the exact width and height of the authentic image.
  - **Fake images:** Pixel-level mask where non-zero pixels (255) denote tampered/inpainted regions.
- `label` (`ClassLabel`): `0` for real (authentic), `1` for fake (manipulated).
- `label_text` (`string`): `"real"` or `"fake"`.
- `dataset` (`string`): Name of the originating benchmark (e.g. `OpenSDID_Flux.1`, `OpenSDID_SD1.5`, `OpenSDID_SDXL`, `SAGI-D`, `CocoGlide`, `TGIF2_flux1dev`, `IMD2020`, `BR-Gen`).
- `method` (`string`): Specific tampering generator or technique (e.g., `Flux.1`, `SD1.5`, `SDXL`, `GLIDE`, `brushnet_epicrealism`, `Inpainting`, `authentic`).
- `mask_ratio` (`float`): Percentage of pixels modified by tampering (`0.0` for authentic, `0.0005 <= ratio <= 0.95` for localized tampering).

---

## 2. Resource Management: RAM & Disk Safety Architecture

To guarantee the pipeline never exhausts RAM or exceeds local disk thresholds (such as 100GB ceilings):

1. **Bounded Streaming (Zero RAM Blowup):**
   - Previous designs held all images in an uncompressed in-memory list before writing shards, causing OOM failures.
   - The streaming pipeline uses a **bounded memory queue** (`queue.Queue(maxsize=1000)`). Worker threads pause when the buffer is full, strictly constraining resident RAM usage to under ~2GB at all times regardless of dataset scale.
2. **Incremental Shard Serialization:**
   - Samples are flushed directly to compressed Parquet shards every `--batch-size` records (default: 500 samples / ~250MB).
   - Once written, in-memory image buffers are immediately cleared and garbage collected (`gc.collect()`).
3. **Strict Disk Space Ceiling (`--max-disk-gb 100.0`):**
   - The pipeline continuously tracks directory sizes and free filesystem blocks. If total written shards exceed the configured limit or filesystem free space drops below 2GB, the builder automatically halts with a descriptive error.
4. **Zero-Disk Accumulation Mode (`--delete-after-upload`):**
   - When enabled alongside `--upload`, each Parquet shard is committed to Hugging Face as soon as it is generated, and the local file is deleted immediately. Local disk usage remains practically near zero.
5. **Immediate Temporary File Cleanup:**
   - All network streaming sources (`OpenSDIDSource`, `SAGIDSource`, `IMD2020Source`) download archive chunks and individual images into temporary directories wrapped in `try ... finally: shutil.rmtree(...)`, guaranteeing immediate removal.

---

## 3. Checkpointing & Resumption

The pipeline includes first-class support for checkpointing and resuming interrupted ingestion jobs:
- State is atomically saved to `checkpoint.json` in `--output-dir` after every written shard.
- Tracks:
  - `completed_shards`: filename, row count, per-dataset breakdown, and Hub upload status.
  - `source_progress`: verified counts of authentic (`real`) and manipulated (`fake`) samples ingested per source.
  - `total_samples`: cumulative record tally.
- **Resumption (`--resume`):**
  When re-executing `build_merged_dataset.py`, already-satisfied dataset targets are skipped entirely, and only remaining deltas are streamed. Shard indexing seamlessly continues from the last completed shard.

---

## 4. Document-Data Reconciliation (Investigation Findings)

An investigation into the dataset on Hugging Face ([`KhangTruong/Tampered-Images-Combined-Dataset-2`](https://huggingface.co/datasets/KhangTruong/Tampered-Images-Combined-Dataset-2)) revealed the root cause of previous document-data mismatches:

1. **BR-Gen Document-Data Discrepancy:**
   - **Issue:** The repository README listed **BR-Gen** under "Datasets Included" and listed its generators (`BrushNet`, `LaMa`, `MAT`, `PowerPaint`) under "Generators & Methods", but the actual 19 Parquet shards contained **0 rows** of BR-Gen.
   - **Root Cause:** BR-Gen is distributed as split archives (`.z01`...`.z12` on HF / Baidu Netdisk). In live execution without a local `--brgen-dir` archive extraction, `BRGenSource` yielded 0 samples. However, earlier build scripts statically appended initialized source names to the README template instead of inspecting actual generated data.
   - **Fix:** `HubUploader` and `DatasetBuilder` now compute dataset cards dynamically based on **verified ingested records** (`count > 0`), ensuring un-downloadable or empty datasets are never misreported in the documentation.
2. **Orphaned Test Shards:**
   - **Issue:** An orphaned test shard `data/cross_test-00000-of-00001.parquet` (120 rows) lingered alongside the 19 production shards (`cross_test-00000-of-00019.parquet` through `cross_test-00018-of-00019.parquet`, total 9,044 rows).
   - **Fix:** Automated synchronization (`--sync-doc`) cleans up orphaned shards and updates `num_examples` in YAML to match the exact 9,044 verified rows.

---

## 5. Dataset Accessibility & Source Matrix

Below is the comprehensive accessibility and download source matrix for investigated image tampering and manipulation localization benchmarks:

| Dataset | data_source | Status | Access Method / Notes / Difficulties |
| :--- | :--- | :--- | :--- |
| **CASIA V1.0** | [https://github.com/namtpham/casia1groundtruth](https://github.com/namtpham/casia1groundtruth) | ✅ Available (Mirror) | Original CASIA server is offline. Standardized masks and image pairs hosted on GitHub; also mirrored in HiFi-IFDL data bundle. |
| **CASIA V2.0** | [https://github.com/namtpham/casia2groundtruth](https://github.com/namtpham/casia2groundtruth) | ✅ Available (Mirror) | Original server offline. Provided via GitHub & Kaggle (`divg07/casia-20-image-tampering-detection-dataset`) with renaming/mask corrections. |
| **Columbia Gray** | [https://www.ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/AuthSplicedDataSet.htm](https://www.ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/AuthSplicedDataSet.htm) | ⚠️ Form-Gated / Mirror | Columbia DVMM lab requires form agreement. 128x128 uncompressed grayscale BMP blocks; bundled in HiFi-IFDL benchmark package. |
| **NIST16 (Nimble 2016 / NC2016)** | [https://data.nist.gov/od/id/mds2-2410](https://data.nist.gov/od/id/mds2-2410) | ⚠️ Registration-Gated | Hosted on NIST OpenMFC portal (`mfc.nist.gov`); requires account registration and Data Transfer Agreement. Deduplication tools on GitHub. |
| **Coverage** | [https://github.com/wenbihan/coverage](https://github.com/wenbihan/coverage) | ✅ Fully Accessible | Official repository for COVERAGE copy-move benchmark (100 image pairs). Ground truth masks provided in MAT/TIFF format. |
| **IMD2020_Inpainting** | [https://huggingface.co/datasets/KhangTruong/IMD2020](https://huggingface.co/datasets/KhangTruong/IMD2020) | ✅ Fully Accessible (HF Hub) | 35,000 synthetic inpainting images with localized binary masks. Streamable directly via Hugging Face Hub Parquet shards (`src/sources/imd2020.py`). |
| **IMD2020_RealLife** | [http://staff.utia.cas.cz/novozada/db/](http://staff.utia.cas.cz/novozada/db/) | ✅ Accessible (Author Web / Bundle) | 2,010 real-world manipulated images collected from the internet with matching pristine sources and manual masks; also in HiFi-IFDL bundle. |
| **TampCOCO** | [https://github.com/mjkwon2021/CAT-Net](https://github.com/mjkwon2021/CAT-Net) | ✅ Accessible (Kaggle / Baidu) | Introduced in CAT-Net for JPEG compression artifact tracing on MS-COCO. Available on Kaggle (`cauchycomplete/cat-net-tampcoco-image-splicing-dataset`). |
| **COCO-Inpaint** | None | ❌ Not Yet Released | Benchmark introduced in arXiv:2504.18361 (Yan et al., April 2025). The authors stated the 238K dataset will be released, but no public download URL is active yet. (Alternative: `KhangTruong/COCO-inpainted` on HF). |
| **COCO-Glide** | [https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip](https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip) | ✅ Fully Automated (Range Stream) | TruFor CVPR 2023 benchmark. 512 GLIDE inpainting pairs on MS-COCO backgrounds. Directly streamable over HTTP without full download. |
| **AutoSplice** | [https://github.com/shanface33/AutoSplice_Dataset](https://github.com/shanface33/AutoSplice_Dataset) | ⚠️ Drive / Baidu Gated | CVPRW 2023 DALL-E 2 prompt-manipulated dataset. Download links provided via Google Drive and Baidu Wangpan in GitHub README. |
| **HiFi_IFDL** | [https://github.com/CHELSEA234/HiFi_IFDL](https://github.com/CHELSEA234/HiFi_IFDL) | ✅ Accessible (Google Drive) | CVPR 2023 / IJCV 2024 hierarchical forgery benchmark. Dataset and bundled standard benchmarks (Columbia, Coverage, CASIA, NIST16, IMD2020) hosted on Google Drive. |
| **OpenSDID (Flux.1 / SD1.5 / SDXL)** | [https://huggingface.co/datasets/nebula/OpenSDI_test](https://huggingface.co/datasets/nebula/OpenSDI_test) | ✅ Fully Automated (Streamed) | arXiv:2503.19653. Streamed row-group by row-group from Hugging Face Hub; filtered for strictly localized inpaintings. |
| **SAGI-D (DiQuID)** | [https://www.kaggle.com/datasets/giakop/sagi-d](https://www.kaggle.com/datasets/giakop/sagi-d) | ✅ Automated via Kaggle API | arXiv:2502.06593. Streamed per-sample using Kaggle API credentials; filtered for spliced (`type == 'sp'`) inpaintings. |
| **TGIF & TGIF2** | [https://github.com/IDLabMedia/tgif-dataset](https://github.com/IDLabMedia/tgif-dataset) | ✅ Fully Automated (Streamed) | WIFS 2024 / JIS 2026. Metadata on GitHub + MS-COCO image retrieval; restricted to spliced (`-sp`) subsets. |
| **BR-Gen** | [https://huggingface.co/datasets/clpbc/BR-Gen](https://huggingface.co/datasets/clpbc/BR-Gen) | ⚡ Requires Local Extraction | AAAI 2026. Distributed as multi-part split archives (`.z01`..`.z12`). Requires manual unpacking into local directory (`--brgen-dir`). |

---

## 6. Quickstart & Usage

### 1. Automated Dataset Downloading (`download_datasets.py`)
Download and unpack any benchmark dataset directly with disk safety controls (automatic archive cleanup, 100GB ceiling):
```bash
# List all supported downloadable benchmarks
python3 download_datasets.py --list

# Download the standard 5-dataset benchmark bundle (Columbia, Coverage, CASIA1, CASIA2, NIST16, IMD2020):
python3 download_datasets.py --datasets standard_bundle

# Download individual datasets:
python3 download_datasets.py --datasets casia1 coverage columbia autosplice

# Download to custom folder with strict 50GB disk ceiling:
python3 download_datasets.py --datasets cocoglide imd2020_inpainting --target-dir /data/datasets --max-disk-gb 50
```

### 2. Mock Test (Zero Bandwidth Dry Run with Checkpoint)
Runs synthetic mock generators across all 17 dataset sources to verify schema, Parquet serialization, and checkpointing:
```bash
python3 build_merged_dataset.py --mode mock --limit 10
```

### 3. Live Run with IMD2020 (HF Hub), COCO-Inpaint, and CocoGlide (Zero Disk Accumulation)
Streams real inpainting pairs directly and uploads to Hugging Face, deleting local shards as soon as they are uploaded:
```bash
python3 build_merged_dataset.py --mode live --datasets imd2020 cocoinpaint cocoglide --limit 100 --upload --delete-after-upload
```

### 4. Live Run with Downloaded Benchmarks (CASIA, Coverage, Columbia, NIST16)
Ingests locally prepared or on-demand downloaded benchmark datasets:
```bash
python3 build_merged_dataset.py --mode live --datasets casia1 columbia coverage nist16 --limit 500
```

### 5. Production Ingestion with Checkpoint Resumption
Streams up to 1,000 real and 1,000 fake images per source with strict 100GB disk ceiling and automatic checkpointing:
```bash
python3 build_merged_dataset.py --mode live --limit 1000 --max-disk-gb 100 --resume --upload
```

### 6. Synchronize Remote Dataset Documentation
Inspects the remote Hugging Face repository, cleans up any orphaned shards, and updates the README dataset card with true row counts and accessibility metadata:
```bash
python3 build_merged_dataset.py --sync-doc
```

---

## 7. Remote Dataset Hub
The benchmark dataset is hosted at:
**[https://huggingface.co/datasets/KhangTruong/Tampered-Images-Combined-Dataset-2](https://huggingface.co/datasets/KhangTruong/Tampered-Images-Combined-Dataset-2)**

