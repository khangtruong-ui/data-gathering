"""
Dataset source loader for HiFi-IFDL (Hierarchical Fine-Grained Image Forgery Detection and Localization).
Guo et al., CVPR 2023 / IJCV 2024.
"""

import os
from typing import Iterator, Optional

from ..downloader import download_dataset
from ..utils import calculate_mask_ratio, create_black_mask_png, get_image_dimensions
from .base import BaseDatasetSource, DataRecord


class HiFiIFDLSource(BaseDatasetSource):
    LOCALIZED_ARCHIVES = ("FaShifter.tar.gz", "STGAN.tar.gz")

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = (
            data_dir
            or os.environ.get("HIFI_IFDL_DIR")
            or os.path.join(os.getcwd(), "data", "HiFi_IFDL")
            or os.path.join(os.getcwd(), "data", "hifi_ifdl")
        )

    @property
    def name(self) -> str:
        return "HiFi_IFDL"

    def _find_local_archives(self) -> dict:
        """Finds localized HiFi-IFDL tar.gz archives in cwd, data_dir, or data/."""
        search_dirs = [
            os.getcwd(),
            self.data_dir,
            os.path.join(os.getcwd(), "data"),
            "/content/HiFi_Dataset",
        ]
        found = {}
        for archive_name in self.LOCALIZED_ARCHIVES:
            for d in search_dirs:
                if not d:
                    continue
                candidate = os.path.join(d, archive_name)
                if os.path.isfile(candidate):
                    found[archive_name] = candidate
                    break
        return found

    def _ensure_data_available(self) -> bool:
        if self._find_local_archives():
            return True
        if os.path.exists(self.data_dir) and len(os.listdir(self.data_dir)) > 0:
            return True
        try:
            print(f"[{self.name}] Data not found at '{self.data_dir}'. Triggering automated download...")
            downloaded = download_dataset("hifi_ifdl", target_dir="data")
            if os.path.exists(downloaded):
                self.data_dir = downloaded
                return True
        except Exception as e:
            print(f"[{self.name}] Automated download failed ({e}).")
        return False

    def _stream_fashifter_tar(self, archive_path: str, limit: int) -> Iterator[DataRecord]:
        """Streams localized FaceShifter image-mask pairs directly from FaShifter.tar.gz."""
        import tarfile

        if limit <= 0:
            return

        print(f"[{self.name}] Streaming up to {limit} localized records from {archive_path}...")
        yielded = 0
        pending_imgs = {}
        pending_masks = {}

        with tarfile.open(archive_path, "r|gz") as tf:
            for member in tf:
                if yielded >= limit:
                    break
                if not member.isfile() or not member.name.endswith(".png"):
                    continue

                bn = member.name.rsplit("/", 1)[-1]
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                raw_bytes = fobj.read()

                if bn.endswith("_mask.png"):
                    key = bn[: -len("_mask.png")]
                    if key in pending_imgs:
                        img_bytes = pending_imgs.pop(key)
                        mask_bytes = raw_bytes
                    else:
                        if len(pending_masks) < limit * 6:
                            pending_masks[key] = raw_bytes
                        continue
                else:
                    key = bn[: -len(".png")]
                    if key in pending_masks:
                        img_bytes = raw_bytes
                        mask_bytes = pending_masks.pop(key)
                    else:
                        if len(pending_imgs) < limit * 6:
                            pending_imgs[key] = raw_bytes
                        continue

                ratio = calculate_mask_ratio(mask_bytes)
                if 0.0005 <= ratio <= 0.95:
                    yield DataRecord(
                        id=f"hifi_fashifter_{key}",
                        image_bytes=img_bytes,
                        image_path=f"{key}.png",
                        mask_bytes=mask_bytes,
                        mask_path=f"{key}_mask.png",
                        label=1,
                        label_text="fake",
                        dataset="HiFi_IFDL",
                        method="FaShifter",
                        mask_ratio=ratio,
                    )
                    yielded += 1

    def _stream_stgan_tar(self, archive_path: str, limit: int) -> Iterator[DataRecord]:
        """Streams localized STGAN image-mask pairs directly from STGAN.tar.gz."""
        import tarfile

        if limit <= 0:
            return

        print(f"[{self.name}] Streaming up to {limit} localized records from {archive_path}...")
        yielded = 0
        pending_imgs = {}
        max_buffer = max(limit * 6, 6000)

        with tarfile.open(archive_path, "r|gz") as tf:
            for member in tf:
                if yielded >= limit:
                    break
                if member.name == "STGAN/dataset_new.zip":
                    break
                if not member.isfile():
                    continue

                if member.name.startswith("STGAN/fake/") and member.name.endswith(".jpg"):
                    if len(pending_imgs) < max_buffer:
                        key = member.name.rsplit("/", 1)[-1][:-4]
                        fobj = tf.extractfile(member)
                        if fobj is not None:
                            pending_imgs[key] = fobj.read()
                elif member.name.startswith("STGAN/mask/") and member.name.endswith("_label.jpg"):
                    key = member.name.rsplit("/", 1)[-1][: -len("_label.jpg")]
                    if key in pending_imgs:
                        fobj = tf.extractfile(member)
                        if fobj is None:
                            continue
                        mask_bytes = fobj.read()
                        img_bytes = pending_imgs.pop(key)
                        ratio = calculate_mask_ratio(mask_bytes)
                        if 0.0005 <= ratio <= 0.95:
                            yield DataRecord(
                                id=f"hifi_stgan_{key}",
                                image_bytes=img_bytes,
                                image_path=f"{key}.jpg",
                                mask_bytes=mask_bytes,
                                mask_path=f"{key}_label.jpg",
                                label=1,
                                label_text="fake",
                                dataset="HiFi_IFDL",
                                method="STGAN",
                                mask_ratio=ratio,
                            )
                            yielded += 1

    def stream_records(
        self,
        target_real: int = 1000,
        target_fake: int = 1000,
    ) -> Iterator[DataRecord]:
        if not self._ensure_data_available():
            print(f"[{self.name}] Notice: Data directory '{self.data_dir}' not available. Skipping.")
            return

        yielded_fake = 0
        yielded_real = 0

        # 1. Stream directly from local tar.gz archives if available (disk-safe, zero extraction overhead)
        archives = self._find_local_archives()
        if archives and target_fake > 0:
            has_stgan = "STGAN.tar.gz" in archives
            has_fashifter = "FaShifter.tar.gz" in archives

            if has_fashifter:
                fa_target = (target_fake + 1) // 2 if has_stgan else target_fake
                for rec in self._stream_fashifter_tar(archives["FaShifter.tar.gz"], fa_target):
                    yield rec
                    yielded_fake += 1
                    if yielded_fake >= target_fake:
                        break

            if has_stgan and yielded_fake < target_fake:
                st_target = target_fake - yielded_fake
                for rec in self._stream_stgan_tar(archives["STGAN.tar.gz"], st_target):
                    yield rec
                    yielded_fake += 1
                    if yielded_fake >= target_fake:
                        break

            # Top up from FaShifter if STGAN yielded fewer than its half
            if has_fashifter and yielded_fake < target_fake:
                seen_ids = set()
                for rec in self._stream_fashifter_tar(archives["FaShifter.tar.gz"], target_fake):
                    if rec.id in seen_ids:
                        continue
                    seen_ids.add(rec.id)
                    if len(seen_ids) <= (target_fake + 1) // 2:
                        continue
                    yield rec
                    yielded_fake += 1
                    if yielded_fake >= target_fake:
                        break

            print(f"[{self.name}] Completed from archives: {yielded_real} real, {yielded_fake} fake.")
            return

        # 2. Fallback: Scan unpacked directory
        print(f"[{self.name}] Scanning directory {self.data_dir}...")
        img_candidates = []
        for root, _, files in os.walk(self.data_dir):
            if "mask" in root.lower():
                continue
            for f in sorted(files):
                if f.lower().endswith((".jpg", ".png", ".jpeg")) and not f.lower().endswith(("_mask.png", "_label.jpg", "_gt.png")):
                    img_candidates.append(os.path.join(root, f))

        for img_path in img_candidates:
            if yielded_fake >= target_fake and yielded_real >= target_real:
                break

            fname = os.path.basename(img_path)
            base_name, _ = os.path.splitext(fname)

            # Search for corresponding mask
            mask_path = None
            parent = os.path.dirname(img_path)
            mask_dirs = [
                parent,
                os.path.join(parent, "mask"),
                os.path.join(os.path.dirname(parent), "mask"),
                os.path.join(self.data_dir, "mask"),
            ]
            for md in mask_dirs:
                if os.path.exists(md):
                    for cand in [f"{base_name}_mask.png", f"{base_name}_label.jpg", f"{base_name}.png", f"{base_name}_gt.png"]:
                        fp = os.path.join(md, cand)
                        if os.path.exists(fp) and fp != img_path:
                            mask_path = fp
                            break
                if mask_path:
                    break

            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()

                if mask_path and os.path.exists(mask_path) and yielded_fake < target_fake:
                    with open(mask_path, "rb") as f:
                        mask_bytes = f.read()

                    ratio = calculate_mask_ratio(mask_bytes)
                    if 0.0005 <= ratio <= 0.95:
                        yield DataRecord(
                            id=f"hifi_{base_name}",
                            image_bytes=img_bytes,
                            image_path=fname,
                            mask_bytes=mask_bytes,
                            mask_path=os.path.basename(mask_path),
                            label=1,
                            label_text="fake",
                            dataset="HiFi_IFDL",
                            method="Hierarchical_Forgery",
                            mask_ratio=ratio,
                        )
                        yielded_fake += 1

                elif "authentic" in img_path.lower() and yielded_real < target_real:
                    w, h = get_image_dimensions(img_bytes)
                    black_mask = create_black_mask_png(w, h)

                    yield DataRecord(
                        id=f"hifi_real_{base_name}",
                        image_bytes=img_bytes,
                        image_path=fname,
                        mask_bytes=black_mask,
                        mask_path=f"real_{base_name}_mask.png",
                        label=0,
                        label_text="real",
                        dataset="HiFi_IFDL",
                        method="authentic",
                        mask_ratio=0.0,
                    )
                    yielded_real += 1
            except Exception:
                continue

        print(f"[{self.name}] Completed: {yielded_real} real, {yielded_fake} fake.")
