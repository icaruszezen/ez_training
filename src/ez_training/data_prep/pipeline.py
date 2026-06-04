"""训练前数据准备主流程。"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
import re
import shutil
from pathlib import Path
from threading import local
from time import perf_counter
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import yaml
from PIL import Image

from ez_training.common.constants import SUPPORTED_IMAGE_FORMATS
from ez_training.data_prep.augmentation import apply_augmentation, build_augmenter
from ez_training.data_prep.converter import (
    build_class_names,
    clear_voc_cache,
    find_voc_for_image,
    load_existing_classes,
    parse_voc_boxes,
    read_voc_image_size,
    save_classes,
    write_voc_annotation,
    write_yolo_label,
)
from ez_training.data_prep.models import (
    EXPORT_FORMAT_VOC,
    EXPORT_FORMAT_YOLO,
    IMAGE_EXPORT_RULE_EXCLUDE_IF_ANY_UNSELECTED,
    DataPrepConfig,
    DataPrepSummary,
    DatasetSample,
    load_custom_class_names,
)
from ez_training.data_prep.splitter import split_train_val


@dataclass
class _PreparedOutputDirs:
    yolo_image_train_dir: Optional[Path] = None
    yolo_image_val_dir: Optional[Path] = None
    yolo_label_train_dir: Optional[Path] = None
    yolo_label_val_dir: Optional[Path] = None
    voc_image_dir: Optional[Path] = None
    voc_annotation_dir: Optional[Path] = None
    voc_image_set_dir: Optional[Path] = None


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
        started_at = perf_counter()
        self.config.validate()
        self._log(log_callback, f"开始数据准备: {self.config.dataset_name}")
        self._log(
            log_callback,
            f"输出格式: {'Pascal VOC' if self.config.export_format == EXPORT_FORMAT_VOC else 'YOLO'}",
        )

        dataset_roots = []
        if self.config.dataset_dirs:
            for d in self.config.dataset_dirs:
                p = Path(d)
                if not p.exists():
                    self._log(log_callback, f"[警告] 目录不存在，跳过: {p}")
                else:
                    dataset_roots.append(p)
        else:
            dataset_root_single = Path(self.config.dataset_dir)
            if not dataset_root_single.exists():
                raise ValueError(f"数据集目录不存在: {dataset_root_single}")
            dataset_roots.append(dataset_root_single)

        if not dataset_roots:
            raise ValueError("没有可用的数据集目录")

        output_root = Path(self.config.output_dir)
        resolved_output = output_root.resolve()
        for dr in dataset_roots:
            resolved_dr = dr.resolve()
            if resolved_output == resolved_dr or self._is_subpath(resolved_output, resolved_dr):
                raise ValueError(
                    f"输出目录 ({output_root}) 不能与数据集源目录 ({dr}) 相同或是其子目录，"
                    f"覆盖模式下可能删除源数据。请选择其他输出目录。"
                )

        prepared_dirs = self._prepare_output_dirs(output_root, self.config.overwrite_output)
        if self.config.export_format == EXPORT_FORMAT_VOC:
            image_train_dir = image_val_dir = prepared_dirs.voc_image_dir
            label_train_dir = label_val_dir = prepared_dirs.voc_annotation_dir
        else:
            image_train_dir = prepared_dirs.yolo_image_train_dir
            image_val_dir = prepared_dirs.yolo_image_val_dir
            label_train_dir = prepared_dirs.yolo_label_train_dir
            label_val_dir = prepared_dirs.yolo_label_val_dir

        if (
            image_train_dir is None
            or image_val_dir is None
            or label_train_dir is None
            or label_val_dir is None
        ):
            raise RuntimeError("输出目录初始化失败")

        scan_started_at = perf_counter()
        all_samples = []
        total_source = 0
        total_skipped = 0
        for dr in dataset_roots:
            s, src, skp = self._scan_samples(dr, log_callback, is_cancelled)
            all_samples.extend(s)
            total_source += src
            total_skipped += skp
        samples = all_samples
        source_images = total_source
        skipped_images = total_skipped
        self._log(log_callback, f"扫描耗时: {perf_counter() - scan_started_at:.2f}s")
        if not samples:
            raise ValueError("没有可处理样本，请确认目录下存在图片和 VOC XML 标注")

        if self.config.custom_classes_file:
            custom_path = Path(self.config.custom_classes_file)
            self._log(log_callback, f"使用自定义类别文件: {custom_path}")
            class_names = load_custom_class_names(custom_path)
            if not class_names:
                raise ValueError(f"自定义类别文件为空: {custom_path}")
            self._log(log_callback, f"自定义类别 ({len(class_names)}): {class_names}")
            class_set = set(class_names)
            selected_class_names = list(self.config.selected_classes)
            selected_class_set = set(selected_class_names)
            self._log(
                log_callback,
                f"已选导出类别 ({len(selected_class_names)}/{len(class_names)}): {selected_class_names}",
            )
            self._log(
                log_callback,
                "整图导出规则: 含未选类别则整图不导出"
                if self.config.image_export_rule
                == IMAGE_EXPORT_RULE_EXCLUDE_IF_ANY_UNSELECTED
                else "整图导出规则: 只要有选择的类别就导出",
            )
            unknown_labels = set()
            for sample in samples:
                for box in sample.boxes:
                    if box.label not in class_set:
                        unknown_labels.add(box.label)
            if unknown_labels:
                self._log(
                    log_callback,
                    f"[警告] 以下标签不在自定义类别中，对应标注框将被跳过: {sorted(unknown_labels)}",
                )
            all_labels = {box.label for s in samples for box in s.boxes}
            if all_labels and not (all_labels & class_set):
                raise ValueError(
                    f"自定义类别 {class_names} 与数据中的实际标签 {sorted(all_labels)} "
                    f"完全不匹配，导出将产生全空标注。请检查自定义类别文件。"
                )
            samples, excluded_by_rule, trimmed_box_images = self._filter_samples_by_selected_classes(
                samples,
                selected_class_set,
            )
            if excluded_by_rule:
                self._log(log_callback, f"按类别规则排除图片: {excluded_by_rule} 张")
            if trimmed_box_images:
                self._log(log_callback, f"导出时移除了未选类别标注的图片: {trimmed_box_images} 张")
            if not samples:
                raise ValueError(
                    "按所选类别和导出规则过滤后，没有可导出的样本。"
                    "请检查勾选的类别或调整整图导出规则。"
                )
        else:
            merged_existing: List[str] = []
            seen: Set[str] = set()
            for dr in dataset_roots:
                for cls in load_existing_classes(dr):
                    if cls not in seen:
                        seen.add(cls)
                        merged_existing.append(cls)
            class_names = build_class_names(samples, merged_existing)
            if not class_names:
                raise ValueError("未能从数据中提取到类别，请确认 VOC 标注有效")
        class_to_id = {name: i for i, name in enumerate(class_names)}

        train_samples, val_samples = split_train_val(
            samples, self.config.train_ratio, self.config.random_seed, dataset_roots
        )
        self._log(
            log_callback,
            f"划分完成: train={len(train_samples)}, val={len(val_samples)}",
        )

        augment_workers = 1
        aug_enabled = bool(self.config.augment_methods) and self.config.augment_times > 0
        if aug_enabled:
            if build_augmenter(self.config.augment_methods) is None:
                aug_enabled = False
                self._log(log_callback, "指定的增强方法均无效，已跳过增强")
            else:
                augment_workers = self._resolve_augment_workers()
                self._log(
                    log_callback,
                    f"启用增强: {', '.join(self.config.augment_methods)} x{self.config.augment_times}",
                )
                self._log(log_callback, f"增强线程数: {augment_workers}")

        total_steps = self._estimate_total_steps(len(train_samples), len(val_samples), aug_enabled)
        done_steps = 0

        used_train_names: Set[str] = set()
        used_val_names: Set[str] = (
            used_train_names if self.config.export_format == EXPORT_FORMAT_VOC else set()
        )
        train_exported_names: List[str] = []
        val_exported_names: List[str] = []

        summary = DataPrepSummary(
            dataset_name=self.config.dataset_name,
            output_dir=str(output_root),
            export_format=self.config.export_format,
            source_images=source_images,
            skipped_images=skipped_images,
            classes_count=len(class_names),
        )

        export_started = False
        try:
            export_started = True
            train_export_started_at = perf_counter()
            summary.train_images, summary.augmented_images, done_steps = self._export_split(
                split_name="train",
                samples=train_samples,
                image_dir=image_train_dir,
                label_dir=label_train_dir,
                used_names=used_train_names,
                dataset_roots=dataset_roots,
                export_format=self.config.export_format,
                class_to_id=class_to_id,
                exported_names=train_exported_names,
                do_augment=aug_enabled,
                progress_total=total_steps,
                progress_done=done_steps,
                progress_callback=progress_callback,
                log_callback=log_callback,
                is_cancelled=is_cancelled,
                augment_workers=augment_workers,
            )
            self._log(log_callback, f"train 导出耗时: {perf_counter() - train_export_started_at:.2f}s")

            val_export_started_at = perf_counter()
            val_images, val_aug_count, done_steps = self._export_split(
                split_name="val",
                samples=val_samples,
                image_dir=image_val_dir,
                label_dir=label_val_dir,
                used_names=used_val_names,
                dataset_roots=dataset_roots,
                export_format=self.config.export_format,
                class_to_id=class_to_id,
                exported_names=val_exported_names,
                do_augment=aug_enabled and self.config.augment_scope == "both",
                progress_total=total_steps,
                progress_done=done_steps,
                progress_callback=progress_callback,
                log_callback=log_callback,
                is_cancelled=is_cancelled,
                augment_workers=augment_workers,
            )
            self._log(log_callback, f"val 导出耗时: {perf_counter() - val_export_started_at:.2f}s")
            summary.val_images = val_images
            summary.augmented_images += val_aug_count
        except BaseException:
            if export_started:
                self._log(log_callback, "[清理] 导出中断，正在清理不完整的输出...")
                self._cleanup_partial_output(output_root)
            clear_voc_cache()
            raise

        classes_path = output_root / "classes.txt"
        save_classes(classes_path, class_names)

        summary.processed_images = summary.train_images + summary.val_images
        summary.classes_path = str(classes_path)
        if self.config.export_format == EXPORT_FORMAT_YOLO:
            yaml_path = output_root / "data.yaml"
            self._save_data_yaml(yaml_path, output_root, class_names)
            summary.yaml_path = str(yaml_path)
            self._log(log_callback, f"已生成 data.yaml: {yaml_path}")
        else:
            if prepared_dirs.voc_image_set_dir is None:
                raise RuntimeError("VOC 索引目录初始化失败")
            train_list_path, val_list_path, trainval_list_path = self._save_voc_image_sets(
                prepared_dirs.voc_image_set_dir,
                train_exported_names,
                val_exported_names,
            )
            summary.train_list_path = str(train_list_path)
            summary.val_list_path = str(val_list_path)
            summary.trainval_list_path = str(trainval_list_path)
            self._log(
                log_callback,
                f"已生成 VOC 划分文件: train={train_list_path}, val={val_list_path}",
            )

        clear_voc_cache()

        self._emit_progress(
            progress_callback, 100, f"完成，导出 {summary.processed_images} 张图片"
        )
        self._log(log_callback, f"输出目录: {output_root}")
        self._log(log_callback, f"总耗时: {perf_counter() - started_at:.2f}s")
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

        seen_xml_map: Dict[str, Path] = {}
        for path in image_paths:
            if is_cancelled and is_cancelled():
                raise RuntimeError("任务已取消")

            xml_path = find_voc_for_image(path, dataset_root, seen_xml_map)
            boxes = []
            width: Optional[int] = None
            height: Optional[int] = None
            mode: Optional[str] = None
            if xml_path is not None:
                try:
                    xml_size = read_voc_image_size(xml_path)
                    if xml_size is not None:
                        width, height = xml_size
                    else:
                        with Image.open(path) as img:
                            width, height = img.size
                            mode = img.mode
                    boxes = parse_voc_boxes(xml_path, width, height)
                except Exception as e:
                    self._log(log_callback, f"[跳过] 标注解析失败 {xml_path.name}: {e}")
                    skipped_images += 1
                    continue

            if self.config.skip_unlabeled and not boxes:
                skipped_images += 1
                continue

            samples.append(
                DatasetSample(
                    image_path=path,
                    xml_path=xml_path,
                    boxes=boxes,
                    image_width=width,
                    image_height=height,
                    image_mode=mode,
                )
            )

        self._log(log_callback, f"有效样本: {len(samples)}，跳过: {skipped_images}")
        return samples, source_images, skipped_images

    def _prepare_output_dirs(
        self, output_root: Path, overwrite_output: bool
    ) -> _PreparedOutputDirs:
        managed_paths = [
            output_root / "images",
            output_root / "labels",
            output_root / "JPEGImages",
            output_root / "Annotations",
            output_root / "ImageSets",
            output_root / "classes.txt",
            output_root / "data.yaml",
        ]
        existing_outputs = [
            path.relative_to(output_root).as_posix()
            for path in managed_paths
            if path.exists()
        ]

        if output_root.exists() and overwrite_output:
            for path in managed_paths:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                elif path.exists():
                    path.unlink(missing_ok=True)

        if output_root.exists() and not overwrite_output and existing_outputs:
            raise ValueError(
                "输出目录已包含旧结果（"
                + ", ".join(existing_outputs)
                + "），请勾选“覆盖输出目录”或更换目录"
            )

        if self.config.export_format == EXPORT_FORMAT_VOC:
            voc_image_dir = output_root / "JPEGImages"
            voc_annotation_dir = output_root / "Annotations"
            voc_image_set_dir = output_root / "ImageSets" / "Main"
            for d in [voc_image_dir, voc_annotation_dir, voc_image_set_dir]:
                d.mkdir(parents=True, exist_ok=True)
            return _PreparedOutputDirs(
                voc_image_dir=voc_image_dir,
                voc_annotation_dir=voc_annotation_dir,
                voc_image_set_dir=voc_image_set_dir,
            )

        img_train_dir = output_root / "images" / "train"
        img_val_dir = output_root / "images" / "val"
        label_train_dir = output_root / "labels" / "train"
        label_val_dir = output_root / "labels" / "val"
        for d in [img_train_dir, img_val_dir, label_train_dir, label_val_dir]:
            d.mkdir(parents=True, exist_ok=True)

        return _PreparedOutputDirs(
            yolo_image_train_dir=img_train_dir,
            yolo_image_val_dir=img_val_dir,
            yolo_label_train_dir=label_train_dir,
            yolo_label_val_dir=label_val_dir,
        )

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

    def _filter_samples_by_selected_classes(
        self,
        samples: List[DatasetSample],
        selected_class_set: Set[str],
    ) -> Tuple[List[DatasetSample], int, int]:
        filtered_samples: List[DatasetSample] = []
        excluded_images = 0
        trimmed_box_images = 0

        for sample in samples:
            if not sample.boxes:
                filtered_samples.append(sample)
                continue

            selected_boxes = [
                box for box in sample.boxes if box.label in selected_class_set
            ]
            if len(selected_boxes) == len(sample.boxes):
                filtered_samples.append(sample)
                continue

            if not selected_boxes:
                excluded_images += 1
                continue

            if self.config.image_export_rule == IMAGE_EXPORT_RULE_EXCLUDE_IF_ANY_UNSELECTED:
                excluded_images += 1
                continue

            trimmed_box_images += 1
            filtered_samples.append(
                DatasetSample(
                    image_path=sample.image_path,
                    xml_path=sample.xml_path,
                    boxes=selected_boxes,
                    image_width=sample.image_width,
                    image_height=sample.image_height,
                    image_mode=sample.image_mode,
                )
            )

        return filtered_samples, excluded_images, trimmed_box_images

    def _export_split(
        self,
        split_name: str,
        samples: List[DatasetSample],
        image_dir: Path,
        label_dir: Path,
        used_names: Set[str],
        dataset_roots: List[Path],
        export_format: str,
        class_to_id: Dict[str, int],
        exported_names: List[str],
        do_augment: bool,
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
        augmenter_cache: Dict[Tuple[int, int], object] = {}

        if do_augment and augment_workers > 1:
            executor = ThreadPoolExecutor(max_workers=augment_workers)
            thread_state = local()

        try:
            for sample in samples:
                if is_cancelled and is_cancelled():
                    raise RuntimeError("任务已取消")

                width = sample.image_width
                height = sample.image_height
                depth = self._infer_image_depth(sample.image_mode)

                base_stem = self._unique_stem(
                    self._make_stem(sample.image_path, dataset_roots), used_names
                )
                base_ext = sample.image_path.suffix.lower()
                if base_ext not in SUPPORTED_IMAGE_FORMATS:
                    base_ext = ".jpg"

                out_img = image_dir / f"{base_stem}{base_ext}"
                shutil.copy2(sample.image_path, out_img)

                image_array = None
                need_open = do_augment or width is None or height is None
                if need_open:
                    with Image.open(sample.image_path) as pil_img:
                        if width is None or height is None:
                            width, height = pil_img.size
                        depth = self._infer_image_depth(pil_img.mode)
                        if do_augment:
                            image_array = np.array(pil_img.convert("RGB"))

                if width is None or height is None:
                    raise ValueError(f"无法获取图片尺寸: {sample.image_path}")

                if export_format == EXPORT_FORMAT_VOC:
                    out_lbl = label_dir / f"{base_stem}.xml"
                    write_voc_annotation(
                        out_lbl,
                        out_img,
                        sample.boxes,
                        width,
                        height,
                        depth=depth,
                    )
                else:
                    out_lbl = label_dir / f"{base_stem}.txt"
                    write_yolo_label(out_lbl, sample.boxes, class_to_id, width, height)

                exported_names.append(base_stem)
                output_count += 1

                progress_done += 1
                self._emit_progress(
                    progress_callback,
                    int(progress_done / progress_total * 100),
                    f"{split_name}: {output_count} 张",
                )

                if not do_augment:
                    continue

                img_size = (height, width)

                if executor is not None and thread_state is not None:
                    augmented_items = self._augment_image_parallel(
                        image_array=image_array,
                        boxes=sample.boxes,
                        image_size=img_size,
                        executor=executor,
                        thread_state=thread_state,
                        is_cancelled=is_cancelled,
                    )
                else:
                    augmenter = augmenter_cache.get(img_size)
                    if augmenter is None:
                        augmenter = build_augmenter(
                            self.config.augment_methods, image_size=img_size
                        )
                        if augmenter is None:
                            raise RuntimeError("增强器构建失败，请检查增强方法配置")
                        augmenter_cache[img_size] = augmenter
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
                    Image.fromarray(aug_image).save(aug_img_path)
                    h, w = aug_image.shape[:2]
                    if export_format == EXPORT_FORMAT_VOC:
                        aug_lbl_path = label_dir / f"{aug_stem}.xml"
                        write_voc_annotation(
                            aug_lbl_path,
                            aug_img_path,
                            aug_boxes,
                            w,
                            h,
                            depth=3,
                        )
                    else:
                        aug_lbl_path = label_dir / f"{aug_stem}.txt"
                        write_yolo_label(aug_lbl_path, aug_boxes, class_to_id, w, h)

                    exported_names.append(aug_stem)
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
        image_size: Tuple[int, int],
        executor: ThreadPoolExecutor,
        thread_state,
        is_cancelled: Optional[Callable[[], bool]],
    ) -> List[Tuple[np.ndarray, List]]:
        def _worker() -> Tuple[np.ndarray, List]:
            cache = getattr(thread_state, "augmenter_cache", None)
            if cache is None:
                cache = {}
                thread_state.augmenter_cache = cache
            local_augmenter = cache.get(image_size)
            if local_augmenter is None:
                local_augmenter = build_augmenter(
                    self.config.augment_methods, image_size=image_size
                )
                if local_augmenter is None:
                    raise RuntimeError("增强器构建失败，请检查增强方法配置")
                cache[image_size] = local_augmenter
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

    def _make_stem(self, image_path: Path, dataset_roots: List[Path]) -> str:
        rel: Optional[str] = None
        for idx, root in enumerate(dataset_roots):
            try:
                part = image_path.relative_to(root).with_suffix("").as_posix()
                rel = f"r{idx}_{part}" if len(dataset_roots) > 1 else part
                break
            except ValueError:
                continue
        if rel is None:
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
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "nc": len(class_names),
            "names": {i: name for i, name in enumerate(class_names)},
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_config, f, allow_unicode=True, default_flow_style=False)

    def _save_voc_image_sets(
        self,
        image_set_dir: Path,
        train_names: List[str],
        val_names: List[str],
    ) -> Tuple[Path, Path, Path]:
        image_set_dir.mkdir(parents=True, exist_ok=True)
        train_list_path = image_set_dir / "train.txt"
        val_list_path = image_set_dir / "val.txt"
        trainval_list_path = image_set_dir / "trainval.txt"

        self._write_name_list(train_list_path, train_names)
        self._write_name_list(val_list_path, val_names)
        self._write_name_list(trainval_list_path, train_names + val_names)
        return train_list_path, val_list_path, trainval_list_path

    @staticmethod
    def _write_name_list(path: Path, names: List[str]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            if names:
                f.write("\n".join(names) + "\n")

    def _emit_progress(
        self,
        callback: Optional[Callable[[int, str], None]],
        percent: int,
        text: str,
    ) -> None:
        if callback:
            callback(max(0, min(100, percent)), text)

    @staticmethod
    def _cleanup_partial_output(output_root: Path) -> None:
        """清理中断导出产生的不完整文件，保留输出根目录本身。"""
        for sub in ["images", "labels", "JPEGImages", "Annotations", "ImageSets"]:
            target = output_root / sub
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
        for name in ["classes.txt", "data.yaml"]:
            target = output_root / name
            if target.exists():
                target.unlink(missing_ok=True)

    @staticmethod
    def _infer_image_depth(mode: Optional[str]) -> int:
        if not mode:
            return 3
        normalized = mode.upper()
        if normalized == "1":
            return 1
        depth = sum(1 for ch in normalized if ch.isalpha())
        return max(1, depth or 1)

    @staticmethod
    def _is_subpath(child: Path, parent: Path) -> bool:
        try:
            child.relative_to(parent)
            return True
        except ValueError:
            return False

    def _log(self, callback: Optional[Callable[[str], None]], text: str) -> None:
        if callback:
            callback(text)
