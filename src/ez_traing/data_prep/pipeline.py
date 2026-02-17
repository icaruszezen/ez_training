"""训练前数据准备主流程。"""

from concurrent.futures import ThreadPoolExecutor
import os
import re
import shutil
from pathlib import Path
from threading import local
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import yaml
from PIL import Image

from ez_traing.common.constants import SUPPORTED_IMAGE_FORMATS
from ez_traing.data_prep.augmentation import apply_augmentation, build_augmenter
from ez_traing.data_prep.converter import (
    build_class_names,
    find_voc_for_image,
    load_existing_classes,
    parse_voc_boxes,
    save_classes,
    write_yolo_label,
)
from ez_traing.data_prep.models import DataPrepConfig, DataPrepSummary, DatasetSample
from ez_traing.data_prep.splitter import split_train_val


class DataPrepPipeline:
    """训练前数据准备执行器。"""

    def __init__(self, config: DataPrepConfig):
        self.config = config

    def run(
        self,
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> DataPrepSummary:
        self.config.validate()
        self._log(log_callback, f"开始数据准备: {self.config.dataset_name}")

        dataset_root = Path(self.config.dataset_dir)
        if not dataset_root.exists():
            raise ValueError(f"数据集目录不存在: {dataset_root}")

        output_root = Path(self.config.output_dir)
        img_train_dir, img_val_dir, label_train_dir, label_val_dir = self._prepare_output_dirs(
            output_root, self.config.overwrite_output
        )

        samples, source_images, skipped_images = self._scan_samples(
            dataset_root, log_callback, is_cancelled
        )
        if not samples:
            raise ValueError("没有可处理样本，请确认目录下存在图片和 VOC XML 标注")

        existing_classes = load_existing_classes(dataset_root)
        class_names = build_class_names(samples, existing_classes)
        if not class_names:
            raise ValueError("未能从数据中提取到类别，请确认 VOC 标注有效")
        class_to_id = {name: i for i, name in enumerate(class_names)}

        train_samples, val_samples = split_train_val(
            samples, self.config.train_ratio, self.config.random_seed
        )
        self._log(
            log_callback,
            f"划分完成: train={len(train_samples)}, val={len(val_samples)}",
        )

        augmenter = None
        augment_workers = 1
        aug_enabled = bool(self.config.augment_methods) and self.config.augment_times > 0
        if aug_enabled:
            augmenter = build_augmenter(self.config.augment_methods)
            if augmenter is not None:
                augment_workers = self._resolve_augment_workers()
            self._log(
                log_callback,
                f"启用增强: {', '.join(self.config.augment_methods)} x{self.config.augment_times}",
            )
            self._log(log_callback, f"增强线程数: {augment_workers}")

        total_steps = self._estimate_total_steps(len(train_samples), len(val_samples), aug_enabled)
        done_steps = 0

        used_train_names: Set[str] = set()
        used_val_names: Set[str] = set()

        summary = DataPrepSummary(
            dataset_name=self.config.dataset_name,
            output_dir=str(output_root),
            source_images=source_images,
            skipped_images=skipped_images,
            classes_count=len(class_names),
        )

        summary.train_images, summary.augmented_images, done_steps = self._export_split(
            split_name="train",
            samples=train_samples,
            image_dir=img_train_dir,
            label_dir=label_train_dir,
            used_names=used_train_names,
            dataset_root=dataset_root,
            class_to_id=class_to_id,
            augmenter=augmenter,
            progress_total=total_steps,
            progress_done=done_steps,
            progress_callback=progress_callback,
            log_callback=log_callback,
            is_cancelled=is_cancelled,
            augment_workers=augment_workers,
        )

        val_images, val_aug_count, done_steps = self._export_split(
            split_name="val",
            samples=val_samples,
            image_dir=img_val_dir,
            label_dir=label_val_dir,
            used_names=used_val_names,
            dataset_root=dataset_root,
            class_to_id=class_to_id,
            augmenter=augmenter if self.config.augment_scope == "both" else None,
            progress_total=total_steps,
            progress_done=done_steps,
            progress_callback=progress_callback,
            log_callback=log_callback,
            is_cancelled=is_cancelled,
            augment_workers=augment_workers,
        )
        summary.val_images = val_images
        summary.augmented_images += val_aug_count

        classes_path = output_root / "classes.txt"
        save_classes(classes_path, class_names)

        yaml_path = output_root / "data.yaml"
        self._save_data_yaml(yaml_path, output_root, class_names)

        summary.processed_images = summary.train_images + summary.val_images
        summary.yaml_path = str(yaml_path)
        summary.classes_path = str(classes_path)

        self._emit_progress(
            progress_callback, 100, f"完成，导出 {summary.processed_images} 张图片"
        )
        self._log(log_callback, f"输出目录: {output_root}")
        return summary

    def _scan_samples(
        self,
        dataset_root: Path,
        log_callback: Optional[Callable[[str], None]],
        is_cancelled: Optional[Callable[[], bool]],
    ) -> Tuple[List[DatasetSample], int, int]:
        samples: List[DatasetSample] = []
        source_images = 0
        skipped_images = 0

        image_paths: List[Path] = []
        for root, _, files in os.walk(dataset_root):
            for name in files:
                path = Path(root) / name
                if path.suffix.lower() in SUPPORTED_IMAGE_FORMATS:
                    image_paths.append(path)

        image_paths.sort()
        source_images = len(image_paths)
        self._log(log_callback, f"扫描到图片: {source_images} 张")

        for path in image_paths:
            if is_cancelled and is_cancelled():
                raise RuntimeError("任务已取消")

            xml_path = find_voc_for_image(path, dataset_root)
            boxes = []
            if xml_path is not None:
                try:
                    with Image.open(path) as img:
                        width, height = img.size
                    boxes = parse_voc_boxes(xml_path, width, height)
                except Exception as e:
                    self._log(log_callback, f"[跳过] 标注解析失败 {xml_path.name}: {e}")
                    skipped_images += 1
                    continue

            if self.config.skip_unlabeled and not boxes:
                skipped_images += 1
                continue

            samples.append(DatasetSample(image_path=path, xml_path=xml_path, boxes=boxes))

        self._log(log_callback, f"有效样本: {len(samples)}，跳过: {skipped_images}")
        return samples, source_images, skipped_images

    def _prepare_output_dirs(
        self, output_root: Path, overwrite_output: bool
    ) -> Tuple[Path, Path, Path, Path]:
        if output_root.exists() and overwrite_output:
            for path in [
                output_root / "images",
                output_root / "labels",
                output_root / "classes.txt",
                output_root / "data.yaml",
            ]:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                elif path.exists():
                    path.unlink(missing_ok=True)

        if output_root.exists() and not overwrite_output:
            if (output_root / "images").exists() or (output_root / "labels").exists():
                raise ValueError("输出目录已包含 images/labels，请勾选“覆盖输出目录”或更换目录")

        img_train_dir = output_root / "images" / "train"
        img_val_dir = output_root / "images" / "val"
        label_train_dir = output_root / "labels" / "train"
        label_val_dir = output_root / "labels" / "val"
        for d in [img_train_dir, img_val_dir, label_train_dir, label_val_dir]:
            d.mkdir(parents=True, exist_ok=True)

        return img_train_dir, img_val_dir, label_train_dir, label_val_dir

    def _estimate_total_steps(self, train_count: int, val_count: int, aug_enabled: bool) -> int:
        base = train_count + val_count
        if not aug_enabled:
            return max(base, 1)
        aug_target = train_count
        if self.config.augment_scope == "both":
            aug_target += val_count
        return max(base + aug_target * self.config.augment_times, 1)

    def _resolve_augment_workers(self) -> int:
        workers = self.config.augment_workers
        if workers <= 0:
            workers = os.cpu_count() or 1
        return max(1, min(workers, self.config.augment_times))

    def _export_split(
        self,
        split_name: str,
        samples: List[DatasetSample],
        image_dir: Path,
        label_dir: Path,
        used_names: Set[str],
        dataset_root: Path,
        class_to_id: Dict[str, int],
        augmenter,
        progress_total: int,
        progress_done: int,
        progress_callback: Optional[Callable[[int, str], None]],
        log_callback: Optional[Callable[[str], None]],
        is_cancelled: Optional[Callable[[], bool]],
        augment_workers: int = 1,
    ) -> Tuple[int, int, int]:
        output_count = 0
        aug_count = 0
        executor: Optional[ThreadPoolExecutor] = None
        thread_state = None
        if augmenter is not None and augment_workers > 1:
            executor = ThreadPoolExecutor(max_workers=augment_workers)
            thread_state = local()

        try:
            for sample in samples:
                if is_cancelled and is_cancelled():
                    raise RuntimeError("任务已取消")

                with Image.open(sample.image_path) as pil_img:
                    image = pil_img.convert("RGB")
                    image_array = np.array(image)
                    width, height = image.size

                base_stem = self._unique_stem(
                    self._make_stem(sample.image_path, dataset_root), used_names
                )
                base_ext = sample.image_path.suffix.lower()
                if base_ext not in SUPPORTED_IMAGE_FORMATS:
                    base_ext = ".jpg"

                out_img = image_dir / f"{base_stem}{base_ext}"
                out_lbl = label_dir / f"{base_stem}.txt"
                image.save(out_img)
                write_yolo_label(out_lbl, sample.boxes, class_to_id, width, height)
                output_count += 1

                progress_done += 1
                self._emit_progress(
                    progress_callback,
                    int(progress_done / progress_total * 100),
                    f"{split_name}: {output_count} 张",
                )

                if augmenter is None:
                    continue

                if executor is not None and thread_state is not None:
                    augmented_items = self._augment_image_parallel(
                        image_array=image_array,
                        boxes=sample.boxes,
                        executor=executor,
                        thread_state=thread_state,
                        is_cancelled=is_cancelled,
                    )
                else:
                    augmented_items = []
                    for _ in range(self.config.augment_times):
                        if is_cancelled and is_cancelled():
                            raise RuntimeError("任务已取消")
                        augmented_items.append(
                            apply_augmentation(image_array, sample.boxes, augmenter)
                        )

                for idx, (aug_image, aug_boxes) in enumerate(augmented_items):
                    if sample.boxes and not aug_boxes:
                        progress_done += 1
                        continue

                    aug_stem = self._unique_stem(f"{base_stem}_aug{idx + 1}", used_names)
                    aug_img_path = image_dir / f"{aug_stem}.jpg"
                    aug_lbl_path = label_dir / f"{aug_stem}.txt"
                    Image.fromarray(aug_image).save(aug_img_path)
                    h, w = aug_image.shape[:2]
                    write_yolo_label(aug_lbl_path, aug_boxes, class_to_id, w, h)

                    output_count += 1
                    aug_count += 1
                    progress_done += 1
                    self._emit_progress(
                        progress_callback,
                        int(progress_done / progress_total * 100),
                        f"{split_name}: 增强 {aug_count} 张",
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        self._log(log_callback, f"{split_name} 导出完成: {output_count} 张")
        return output_count, aug_count, progress_done

    def _augment_image_parallel(
        self,
        image_array: np.ndarray,
        boxes: List,
        executor: ThreadPoolExecutor,
        thread_state,
        is_cancelled: Optional[Callable[[], bool]],
    ) -> List[Tuple[np.ndarray, List]]:
        def _worker() -> Tuple[np.ndarray, List]:
            local_augmenter = getattr(thread_state, "augmenter", None)
            if local_augmenter is None:
                local_augmenter = build_augmenter(self.config.augment_methods)
                if local_augmenter is None:
                    raise RuntimeError("增强器构建失败，请检查增强方法配置")
                thread_state.augmenter = local_augmenter
            return apply_augmentation(image_array, boxes, local_augmenter)

        futures = [executor.submit(_worker) for _ in range(self.config.augment_times)]
        results: List[Tuple[np.ndarray, List]] = []
        for f in futures:
            if is_cancelled and is_cancelled():
                for pending in futures:
                    pending.cancel()
                raise RuntimeError("任务已取消")
            results.append(f.result())
        return results

    def _make_stem(self, image_path: Path, dataset_root: Path) -> str:
        try:
            rel = image_path.relative_to(dataset_root).with_suffix("").as_posix()
        except ValueError:
            rel = image_path.stem
        rel = rel.replace("/", "_").replace("\\", "_")
        return re.sub(r"[^\w\-]+", "_", rel).strip("_") or "img"

    def _unique_stem(self, stem: str, used_names: Set[str]) -> str:
        if stem not in used_names:
            used_names.add(stem)
            return stem
        idx = 1
        while f"{stem}_{idx}" in used_names:
            idx += 1
        unique = f"{stem}_{idx}"
        used_names.add(unique)
        return unique

    def _save_data_yaml(self, yaml_path: Path, output_root: Path, class_names: List[str]) -> None:
        data_config = {
            "path": str(output_root),
            "train": "images/train",
            "val": "images/val",
            "nc": len(class_names),
            "names": {i: name for i, name in enumerate(class_names)},
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_config, f, allow_unicode=True, default_flow_style=False)

    def _emit_progress(
        self,
        callback: Optional[Callable[[int, str], None]],
        percent: int,
        text: str,
    ) -> None:
        if callback:
            callback(max(0, min(100, percent)), text)

    def _log(self, callback: Optional[Callable[[str], None]], text: str) -> None:
        if callback:
            callback(text)
