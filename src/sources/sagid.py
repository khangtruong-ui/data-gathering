"""
Dataset stream reader for SAGI-D (arXiv:2502.06593).
Automated via the Kaggle API using ~/.kaggle/kaggle.json.
Streams individual images and masks concurrently using ThreadPoolExecutor,
and filters for spliced inpaintings (type == 'sp') to strictly avoid 100% fake images.
Silences Kaggle API URL and download stdout spam.
"""

from concurrent.futures import ThreadPoolExecutor
import csv
import os
import shutil
import tempfile
from typing import Iterator, List, Tuple

from ..utils import calculate_mask_ratio, create_black_mask_png, get_image_dimensions
from .base import BaseDatasetSource, DataRecord


class SAGIDSource(BaseDatasetSource):
    KAGGLE_DATASET = "giakop/sagi-d"
    CSV_FILENAME = "sagid.csv"

    def __init__(self, target_model: str = None):
        """
        target_model: optional filter for inpainting_model (e.g., 'brushnet', 'powerpaint', 'controlnet')
        """
        self.target_model = target_model
        self._api = None

    @property
    def name(self) -> str:
        if self.target_model:
            return f"SAGID_{self.target_model}"
        return "SAGID"

    def _get_api(self):
        if self._api is None:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            # Silence Kaggle API print spam (Dataset URL: https://www.kaggle.com/datasets/...)
            api._print_dataset_url_and_license = lambda *args, **kwargs: None
            self._api = api
        return self._api

    def _download_file(self, rel_path: str, target_dir: str) -> bool:
        """Downloads a single file from Kaggle in a thread-safe manner."""
        api = self._get_api()
        try:
            return api.dataset_download_file(
                self.KAGGLE_DATASET, rel_path, path=target_dir, quiet=True
            )
        except Exception:
            return False

    def _get_csv_path(self) -> str:
        cached_path = "/tmp/sagid.csv"
        if os.path.exists(cached_path) and os.path.getsize(cached_path) > 1000:
            return cached_path

        self._download_file(self.CSV_FILENAME, "/tmp")
        # Kaggle might extract to /tmp/sagid.csv or leave /tmp/sagid.csv.zip
        if not os.path.exists(cached_path) and os.path.exists("/tmp/sagid.csv.zip"):
            import zipfile

            with zipfile.ZipFile("/tmp/sagid.csv.zip", "r") as z:
                z.extractall("/tmp")
        return cached_path

    def _fetch_row_records(
        self,
        row: dict,
        work_dir: str,
        need_fake: bool,
        need_real: bool,
    ) -> List[DataRecord]:
        records = []

        # Clean path strings: replace Windows backslashes and strip 'sagid/' prefix if present
        img_rel = row["img_path"].replace("\\", "/")
        if img_rel.startswith("sagid/"):
            img_rel = img_rel[len("sagid/") :]

        mask_rel = row["mask_path"].replace("\\", "/")
        if mask_rel.startswith("sagid/"):
            mask_rel = mask_rel[len("sagid/") :]

        src_rel = row["src_path"].replace("\\", "/")
        if src_rel.startswith("sagid/"):
            src_rel = src_rel[len("sagid/") :]

        inpaint_m = row.get("inpainting_model", "brushnet")
        diff_m = row.get("diffusion_model", "sdxl")
        method_name = f"{inpaint_m}_{diff_m}"
        base_name = os.path.basename(img_rel).replace(".jpg", "").replace(".png", "")

        # 1. Fake image + mask
        if need_fake:
            self._download_file(img_rel, work_dir)
            self._download_file(mask_rel, work_dir)

            local_img = os.path.join(work_dir, os.path.basename(img_rel))
            local_mask = os.path.join(work_dir, os.path.basename(mask_rel))

            if os.path.exists(local_img) and os.path.exists(local_mask):
                try:
                    with open(local_img, "rb") as f:
                        img_bytes = f.read()
                    with open(local_mask, "rb") as f:
                        mask_bytes = f.read()

                    ratio = calculate_mask_ratio(mask_bytes)
                    if 0.0005 <= ratio <= 0.95:
                        records.append(
                            DataRecord(
                                id=f"sagid_{base_name}",
                                image_bytes=img_bytes,
                                image_path=os.path.basename(img_rel),
                                mask_bytes=mask_bytes,
                                mask_path=os.path.basename(mask_rel),
                                label=1,
                                label_text="fake",
                                dataset="SAGI-D",
                                method=method_name,
                                mask_ratio=ratio,
                            )
                        )
                finally:
                    if os.path.exists(local_img):
                        os.remove(local_img)
                    if os.path.exists(local_mask):
                        os.remove(local_mask)

        # 2. Real original image
        if need_real:
            self._download_file(src_rel, work_dir)
            local_src = os.path.join(work_dir, os.path.basename(src_rel))

            if os.path.exists(local_src):
                try:
                    with open(local_src, "rb") as f:
                        src_bytes = f.read()
                    w, h = get_image_dimensions(src_bytes)
                    black_mask = create_black_mask_png(w, h)

                    src_id = os.path.basename(src_rel).replace(".jpg", "").replace(".png", "")
                    records.append(
                        DataRecord(
                            id=f"sagid_real_{src_id}",
                            image_bytes=src_bytes,
                            image_path=os.path.basename(src_rel),
                            mask_bytes=black_mask,
                            mask_path=f"real_{src_id}_mask.png",
                            label=0,
                            label_text="real",
                            dataset="SAGI-D",
                            method="authentic",
                            mask_ratio=0.0,
                        )
                    )
                finally:
                    if os.path.exists(local_src):
                        os.remove(local_src)

        return records

    def stream_records(
        self,
        target_real: int = 1000,
        target_fake: int = 1000,
    ) -> Iterator[DataRecord]:
        csv_path = self._get_csv_path()

        sp_rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Strictly select spliced ('sp') to avoid 100% fake full-regeneration
                if row.get("type") != "sp":
                    continue
                if self.target_model and row.get("inpainting_model") != self.target_model:
                    continue
                sp_rows.append(row)
                if len(sp_rows) >= (target_fake + target_real) * 2:
                    break

        yielded_real = 0
        yielded_fake = 0

        # Create temporary working directory for on-demand per-file downloads
        work_dir = tempfile.mkdtemp(prefix="sagid_stream_")

        batch_size = 16
        try:
            with ThreadPoolExecutor(max_workers=8) as executor:
                for i in range(0, len(sp_rows), batch_size):
                    if yielded_real >= target_real and yielded_fake >= target_fake:
                        break

                    batch_rows = sp_rows[i : i + batch_size]
                    need_fake = yielded_fake < target_fake
                    need_real = yielded_real < target_real

                    futures = [
                        executor.submit(
                            self._fetch_row_records, row, work_dir, need_fake, need_real
                        )
                        for row in batch_rows
                    ]

                    for fut in futures:
                        for rec in fut.result():
                            if rec.label == 1 and yielded_fake < target_fake:
                                yield rec
                                yielded_fake += 1
                            elif rec.label == 0 and yielded_real < target_real:
                                yield rec
                                yielded_real += 1

                            if yielded_real >= target_real and yielded_fake >= target_fake:
                                break
                        if yielded_real >= target_real and yielded_fake >= target_fake:
                            break
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
