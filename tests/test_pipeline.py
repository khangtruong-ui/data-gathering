"""
Unit and integration tests for dataset pipeline:
- Memory & disk safe streaming builder.
- Checkpoint saving and resumption.
- Accurate dynamic dataset card generation with data_source column.
- Mask ratio computation on PNG and JPEG formats.
"""

import io
import json
import os
import shutil
import tempfile
import unittest

from PIL import Image
import pyarrow.parquet as pq

from src.builder import DatasetBuilder
from src.sources.mock_source import MockDatasetSource
from src.uploader import INVESTIGATED_DATASETS_MATRIX, generate_dataset_card
from src.utils import calculate_mask_ratio, create_black_mask_png


class TestDatasetPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_pipeline_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_calculate_mask_ratio_png_and_jpeg(self):
        # 1. Black mask PNG
        black_png = create_black_mask_png(100, 100)
        self.assertEqual(calculate_mask_ratio(black_png), 0.0)

        # 2. Tampered PNG (50% white)
        img = Image.new("L", (100, 100), 0)
        for y in range(50):
            for x in range(100):
                img.putpixel((x, y), 255)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        ratio_png = calculate_mask_ratio(buf.getvalue())
        self.assertAlmostEqual(ratio_png, 0.5, places=2)

        # 3. JPEG mask with compression noise
        buf_jpg = io.BytesIO()
        img.save(buf_jpg, format="JPEG", quality=80)
        ratio_jpg = calculate_mask_ratio(buf_jpg.getvalue())
        self.assertAlmostEqual(ratio_jpg, 0.5, places=2)

    def test_builder_streaming_and_checkpoint(self):
        builder = DatasetBuilder(output_dir=self.test_dir, max_disk_gb=100.0)
        sources = [
            MockDatasetSource("MockSetA", method="GenA"),
            MockDatasetSource("MockSetB", method="GenB"),
        ]

        # First run: limit 4 real, 4 fake per source (total 16 records)
        # batch_size=5 -> produces 3 shards (5, 5, 6 records)
        shards = builder.build_split(
            sources=sources,
            split_name="test_split",
            target_real_per_dataset=4,
            target_fake_per_dataset=4,
            batch_size=5,
            max_workers=2,
            resume=True,
        )

        self.assertTrue(len(shards) >= 3)
        ckpt_path = builder.get_checkpoint_path()
        self.assertTrue(os.path.exists(ckpt_path))

        with open(ckpt_path, "r") as f:
            ckpt = json.load(f)
        self.assertEqual(ckpt["total_samples"], 16)
        self.assertEqual(ckpt["source_progress"]["MockSetA"]["real"], 4)
        self.assertEqual(ckpt["source_progress"]["MockSetA"]["fake"], 4)

        # Second run: resume with higher limit (6 real, 6 fake per source = 24 records total)
        # It should only generate the remaining 8 records!
        shards_resumed = builder.build_split(
            sources=sources,
            split_name="test_split",
            target_real_per_dataset=6,
            target_fake_per_dataset=6,
            batch_size=5,
            max_workers=2,
            resume=True,
        )

        with open(ckpt_path, "r") as f:
            ckpt2 = json.load(f)
        self.assertEqual(ckpt2["total_samples"], 24)
        self.assertEqual(ckpt2["source_progress"]["MockSetA"]["real"], 6)

    def test_dynamic_dataset_card_generation(self):
        # Card should only include datasets with > 0 samples
        datasets_dist = {"OpenSDID_Flux.1": 2000, "CocoGlide": 1000, "BR-Gen": 0}
        methods_dist = {"Flux.1": 1000, "GLIDE": 500, "authentic": 1500, "PowerPaint": 0}

        card = generate_dataset_card(
            repo_id="TestOrg/TestRepo",
            split_name="cross_test",
            total_samples=3000,
            datasets_distribution=datasets_dist,
            methods_distribution=methods_dist,
        )

        # Check YAML num_examples
        self.assertIn("num_examples: 3000", card)
        # Check active datasets
        self.assertIn("- **CocoGlide**: 1,000 samples", card)
        self.assertIn("- **OpenSDID_Flux.1**: 2,000 samples", card)
        # Check that 0-sample dataset BR-Gen and 0-sample method PowerPaint are NOT in the active present lists
        self.assertNotIn("- **BR-Gen**", card)
        self.assertNotIn("- `PowerPaint`", card)

        # Check that data_source column exists in the matrix
        self.assertIn("| Dataset | data_source | Status | Notes / Difficulties / Access Details |", card)
        # Check that COCO-Inpaint has data_source None
        self.assertIn("| **COCO-Inpaint** | None | ❌ Not Yet Released |", card)
        # Check that CASIA V1.0 has a valid link
        self.assertIn("https://github.com/namtpham/casia1groundtruth", card)
        # Check that IMD2020 has link to KhangTruong/IMD2020
        self.assertIn("https://huggingface.co/datasets/KhangTruong/IMD2020", card)

    def test_downloader_catalog_and_disk_safety(self):
        from src.downloader import DATASET_DOWNLOAD_CATALOG, check_disk_safety

        self.assertIn("standard_bundle", DATASET_DOWNLOAD_CATALOG)
        self.assertIn("casia1", DATASET_DOWNLOAD_CATALOG)
        self.assertIn("casia2", DATASET_DOWNLOAD_CATALOG)
        self.assertIn("columbia", DATASET_DOWNLOAD_CATALOG)
        self.assertIn("coverage", DATASET_DOWNLOAD_CATALOG)
        self.assertIn("nist16", DATASET_DOWNLOAD_CATALOG)
        self.assertIn("autosplice", DATASET_DOWNLOAD_CATALOG)
        self.assertIn("hifi_ifdl", DATASET_DOWNLOAD_CATALOG)

        # Safety check with normal limit passes
        check_disk_safety(self.test_dir, estimated_gb=0.1, max_limit_gb=100.0)

        # Safety check with tight limit raises RuntimeError
        with self.assertRaises(RuntimeError):
            check_disk_safety(self.test_dir, estimated_gb=2000.0, max_limit_gb=100.0)

    def test_sources_interface_instantiation(self):
        from src.sources import (
            AutoSpliceSource,
            CasiaSource,
            CocoInpaintSource,
            ColumbiaSource,
            CoverageSource,
            HiFiIFDLSource,
            IMD2020RealLifeSource,
            IMD2020Source,
            NIST16Source,
        )

        sources = [
            CasiaSource(version="v1", data_dir="/tmp/dummy"),
            CasiaSource(version="v2", data_dir="/tmp/dummy"),
            ColumbiaSource(data_dir="/tmp/dummy"),
            CoverageSource(data_dir="/tmp/dummy"),
            NIST16Source(data_dir="/tmp/dummy"),
            IMD2020Source(split="test"),
            IMD2020RealLifeSource(data_dir="/tmp/dummy"),
            CocoInpaintSource(),
            AutoSpliceSource(data_dir="/tmp/dummy"),
            HiFiIFDLSource(data_dir="/tmp/dummy"),
        ]

        for s in sources:
            self.assertTrue(len(s.name) > 0)
            self.assertTrue(hasattr(s, "stream_records"))


if __name__ == "__main__":
    unittest.main()

