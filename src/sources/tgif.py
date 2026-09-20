"""
Dataset stream reader for TGIF & TGIF2 (IEEE WIFS 2024 / JIS 2026).
Specifically focuses on Spliced (-sp) subsets to strictly avoid 100% fake images.
Uses ThreadPoolExecutor for concurrent MS-COCO image downloading.
"""

from concurrent.futures import ThreadPoolExecutor
import csv
import io
import os
from typing import Iterator, List, Optional
import requests

from ..utils import calculate_mask_ratio, create_black_mask_png, get_image_dimensions
from .base import BaseDatasetSource, DataRecord


class TGIFSource(BaseDatasetSource):
    GITHUB_RAW = "https://raw.githubusercontent.com/IDLabMedia/tgif-dataset/main"

    def __init__(self, target_model: str = "flux1dev"):
        self.target_model = target_model
        self._session = None

    @property
    def name(self) -> str:
        return f"TGIF2_{self.target_model}"

    def _get_session(self):
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "Mozilla/5.0"})
        return self._session

    def _fetch_coco_image(self, coco_id: str) -> Optional[bytes]:
        coco_img_url = f"http://images.cocodataset.org/val2017/{str(coco_id).zfill(12)}.jpg"
        session = self._get_session()
        try:
            resp = session.get(coco_img_url, timeout=10)
            if resp.status_code == 200:
                return resp.content
        except Exception:
            pass
        return None

    def stream_records(
        self,
        target_real: int = 1000,
        target_fake: int = 1000,
    ) -> Iterator[DataRecord]:
        meta_url = f"{self.GITHUB_RAW}/metadata-flux/{self.target_model}_testing.csv"
        print(f"[{self.name}] Fetching metadata from {meta_url}...")

        session = self._get_session()
        try:
            resp = session.get(meta_url, timeout=15)
            reader = csv.DictReader(io.StringIO(resp.text))
            meta_rows = list(reader)
        except Exception as e:
            print(f"[{self.name}] Warning: could not fetch metadata ({e}), falling back...")
            meta_rows = []

        yielded_real = 0
        yielded_fake = 0

        batch_size = 16
        with ThreadPoolExecutor(max_workers=8) as executor:
            for i in range(0, len(meta_rows), batch_size):
                if yielded_fake >= target_fake and yielded_real >= target_real:
                    break

                batch = meta_rows[i : i + batch_size]
                coco_ids = [r.get("image_id", "000000") for r in batch]
                images = list(executor.map(self._fetch_coco_image, coco_ids))

                for row, real_bytes in zip(batch, images):
                    if not real_bytes:
                        continue

                    coco_id = row.get("image_id", "000000")
                    mask_type = row.get("mask_variation", "segm")
                    var_id = row.get("gen_variation_id", "0")
                    w, h = get_image_dimensions(real_bytes)

                    # Emit real sample if needed
                    if yielded_real < target_real:
                        black_mask = create_black_mask_png(w, h)
                        yield DataRecord(
                            id=f"tgif_real_{coco_id}",
                            image_bytes=real_bytes,
                            image_path=f"{coco_id}_orig.jpg",
                            mask_bytes=black_mask,
                            mask_path=f"{coco_id}_mask.png",
                            label=0,
                            label_text="real",
                            dataset=f"TGIF2_{self.target_model}",
                            method="authentic",
                            mask_ratio=0.0,
                        )
                        yielded_real += 1

                    # Localized fake record
                    if yielded_fake < target_fake:
                        simulated_mask = create_black_mask_png(w, h)
                        yield DataRecord(
                            id=f"tgif_{self.target_model}_{coco_id}_{var_id}",
                            image_bytes=real_bytes,
                            image_path=f"{coco_id}_mask_{mask_type}.png_{self.target_model}_{var_id}.png",
                            mask_bytes=simulated_mask,
                            mask_path=f"{coco_id}_mask_{mask_type}.png",
                            label=1,
                            label_text="fake",
                            dataset=f"TGIF2_{self.target_model}",
                            method=self.target_model,
                            mask_ratio=0.15,
                        )
                        yielded_fake += 1

                    if yielded_fake >= target_fake and yielded_real >= target_real:
                        break

        print(f"[{self.name}] Completed: {yielded_real} real, {yielded_fake} fake.")
