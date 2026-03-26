"""YOLO 验证引擎。"""

import logging
import math
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import yaml

from ez_training.common.constants import get_config_dir, SUPPORTED_IMAGE_FORMATS
from ez_training.common.voc_io import parse_voc_objects, parse_voc_size
from ez_training.evaluation.models import EvalConfig, EvalMetrics, EvalResult
from ez_training.evaluation.visualization import discover_yolo_plots, generate_fallback_charts

logger = logging.getLogger(__name__)


# \w matches Unicode letters (including CJK), intentional for Chinese names.
def _sanitize_name(value: str) -> str:
    value = re.sub(r"[^\w\-]+", "_", value.strip())
    return value.strip("_") or "val"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        if not math.isfinite(result):
            return default
        return result
    except Exception:
        return default


def _read_classes(dataset_dir: Path) -> List[str]:
    for path in [dataset_dir / "classes.txt", dataset_dir / "labels" / "classes.txt"]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                names = [line.strip() for line in f if line.strip()]
            if names:
                return names

    for yaml_name in ("data.yaml", "data.yml", "dataset.yaml"):
        yaml_path = dataset_dir / yaml_name
        if yaml_path.exists():
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict) and "names" in data:
                    names_val = data["names"]
                    if isinstance(names_val, dict):
                        return [names_val[k] for k in sorted(names_val.keys())]
                    if isinstance(names_val, list):
                        return [str(n) for n in names_val if str(n).strip()]
            except Exception:
                continue

    raise ValueError(f"找不到 classes.txt 或 data.yaml: {dataset_dir}")


def build_data_yaml(dataset_name: str, dataset_dir: str, output_dir: str) -> str:
    """根据数据集目录生成 YOLO data yaml。"""
    root = Path(dataset_dir)
    if not root.exists():
        raise ValueError(f"数据集目录不存在: {dataset_dir}")

    class_names = _read_classes(root)

    images_dir = root / "images"
    has_images_subdir = images_dir.exists()
    if not has_images_subdir:
        images_dir = root

    train_dir = images_dir / "train"
    val_dir = images_dir / "val"

    if train_dir.exists() and val_dir.exists():
        train_path = train_dir.relative_to(root).as_posix()
        val_path = val_dir.relative_to(root).as_posix()
    elif has_images_subdir:
        train_path = "images"
        val_path = "images"
    else:
        train_path = "."
        val_path = "."

    data_config = {
        "path": str(root),
        "train": train_path,
        "val": val_path,
        "names": {i: name for i, name in enumerate(class_names)},
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    yaml_path = out / f"{_sanitize_name(dataset_name)}_val_data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, allow_unicode=True, default_flow_style=False)
    return str(yaml_path)


def _is_voc_dataset(dataset_dir: str) -> bool:
    """Return True if the directory (or its subdirectories) contains VOC XML files."""
    root = Path(dataset_dir)
    if not root.is_dir():
        return False
    return any(root.rglob("*.xml"))


def _link_or_copy(src: Path, dst: Path) -> None:
    """Create a symlink; fall back to hard link then copy on failure."""
    try:
        os.symlink(src, dst)
        return
    except OSError:
        pass
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    shutil.copy2(src, dst)


def prepare_voc_dataset(
    dataset_dir: str,
    output_dir: str,
    dataset_name: str,
    model_names: Dict[int, str],
    include_unannotated: bool = False,
    log_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Build a temporary YOLO-format dataset from images + VOC XML annotations.

    *model_names* must be the class name dict from the loaded YOLO model
    (``{0: 'cls_a', 1: 'cls_b', ...}``).  The mapping ensures XML labels
    are converted to the exact indices the model expects.

    Returns the path to the generated ``data.yaml``.
    """

    def emit(msg: str) -> None:
        if log_callback:
            log_callback(msg)

    root = Path(dataset_dir)
    if not root.is_dir():
        raise ValueError(f"数据集目录不存在: {dataset_dir}")

    class_to_idx: Dict[str, int] = {name: idx for idx, name in model_names.items()}
    emit(f"[INFO] 使用模型类别 ({len(model_names)}): {list(model_names.values())}")

    image_files: List[Path] = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_FORMATS
    )
    if not image_files:
        raise ValueError(f"数据集目录中未找到图片: {dataset_dir}")

    emit(f"[INFO] 扫描到 {len(image_files)} 张图片")

    pairs: List[Tuple[Path, Optional[Path]]] = []
    annotated = 0
    unannotated = 0
    skipped_labels: set = set()

    for img in image_files:
        xml_path = img.with_suffix(".xml")
        if xml_path.exists():
            objects = parse_voc_objects(xml_path)
            for obj in objects:
                if obj.label not in class_to_idx:
                    skipped_labels.add(obj.label)
            pairs.append((img, xml_path))
            annotated += 1
        elif include_unannotated:
            pairs.append((img, None))
            unannotated += 1

    if not pairs:
        raise ValueError("未找到带 XML 标注的图片（若要包含无标注图片请开启对应选项）")

    if skipped_labels:
        emit(f"[WARN] XML 中存在模型未包含的类别（将跳过）: {sorted(skipped_labels)}")

    emit(f"[INFO] 已标注: {annotated}, 无标注: {unannotated}")

    prep_dir = Path(output_dir) / f"_voc_prepared_{_sanitize_name(dataset_name)}"
    images_dir = prep_dir / "images" / "val"
    labels_dir = prep_dir / "labels" / "val"
    if prep_dir.exists():
        shutil.rmtree(prep_dir, ignore_errors=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    seen_names: Dict[str, int] = {}
    converted = 0

    for img_path, xml_path in pairs:
        rel = img_path.relative_to(root)
        if len(rel.parts) > 1:
            prefix = "_".join(rel.parts[:-1]) + "_"
        else:
            prefix = ""
        stem = prefix + img_path.stem
        suffix = img_path.suffix
        if stem in seen_names:
            seen_names[stem] += 1
            stem = f"{stem}_{seen_names[stem]}"
        else:
            seen_names[stem] = 0

        dst_img = images_dir / f"{stem}{suffix}"
        _link_or_copy(img_path, dst_img)

        label_path = labels_dir / f"{stem}.txt"
        if xml_path is not None:
            objects = parse_voc_objects(xml_path)
            size = parse_voc_size(xml_path)
            if size and objects:
                w, h = size
                lines: List[str] = []
                for obj in objects:
                    if obj.label not in class_to_idx:
                        continue
                    cx = (obj.xmin + obj.xmax) / 2.0 / w
                    cy = (obj.ymin + obj.ymax) / 2.0 / h
                    bw = (obj.xmax - obj.xmin) / w
                    bh = (obj.ymax - obj.ymin) / h
                    lines.append(f"{class_to_idx[obj.label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                label_path.write_text("\n".join(lines), encoding="utf-8")
                converted += 1
            else:
                label_path.write_text("", encoding="utf-8")
        else:
            label_path.write_text("", encoding="utf-8")

    emit(f"[INFO] 已转换 {converted} 个 XML 标注为 YOLO 格式")

    data_config = {
        "path": str(prep_dir),
        "train": "images/val",
        "val": "images/val",
        "names": dict(model_names),
    }
    yaml_path = prep_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, allow_unicode=True, default_flow_style=False)

    emit(f"[INFO] 生成 data yaml: {yaml_path}")
    return str(yaml_path)


class EvaluationEngine:
    """模型验证执行器。"""

    def run(
        self,
        config: EvalConfig,
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> EvalResult:
        def emit_log(text: str):
            if log_callback:
                log_callback(text)

        def emit_progress(value: int):
            if progress_callback:
                progress_callback(max(0, min(100, value)))

        def is_cancelled() -> bool:
            return cancel_check() if cancel_check else False

        _CANCEL_MSG = "验证已取消"
        model = None

        try:
            emit_progress(2)
            if is_cancelled():
                return EvalResult(success=False, message=_CANCEL_MSG)

            model_path = Path(config.model_path)
            if not model_path.exists() or model_path.suffix.lower() != ".pt":
                raise ValueError("请选择存在的 YOLO 权重文件（.pt）")

            output_root = Path(config.output_root) if config.output_root else get_config_dir() / "runs" / "val"
            output_root.mkdir(parents=True, exist_ok=True)
            run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_sanitize_name(config.dataset_name)}_{_sanitize_name(model_path.stem)}"

            emit_log(f"[INFO] 数据集: {config.dataset_name}")
            emit_log(f"[INFO] 数据集目录: {config.dataset_dir}")
            emit_log(f"[INFO] 模型权重: {config.model_path}")
            emit_progress(10)

            if is_cancelled():
                return EvalResult(success=False, message=_CANCEL_MSG)

            try:
                from ultralytics import YOLO
            except ImportError as e:
                raise RuntimeError("未安装 ultralytics，请先安装依赖") from e

            emit_log("[INFO] 正在加载模型...")
            model = YOLO(config.model_path)
            model_names = getattr(model, "names", None) or {}
            emit_log(f"[INFO] 模型类别 ({len(model_names)}): {list(model_names.values())}")
            emit_progress(20)

            if is_cancelled():
                return EvalResult(success=False, message=_CANCEL_MSG)

            if _is_voc_dataset(config.dataset_dir):
                emit_log("[INFO] 检测到 VOC 格式数据集，正在转换为 YOLO 格式...")
                if not model_names:
                    raise ValueError("模型中未包含类别信息，无法进行 VOC 数据集转换")
                data_yaml = prepare_voc_dataset(
                    dataset_dir=config.dataset_dir,
                    output_dir=str(output_root),
                    dataset_name=config.dataset_name,
                    model_names=model_names,
                    include_unannotated=config.include_unannotated,
                    log_callback=log_callback,
                )
            else:
                try:
                    data_yaml = build_data_yaml(config.dataset_name, config.dataset_dir, str(output_root))
                    emit_log(f"[INFO] 生成 data yaml: {data_yaml}")
                except ValueError:
                    emit_log("[WARN] YOLO 格式加载失败，尝试使用 VOC XML 标注...")
                    if not model_names:
                        raise
                    data_yaml = prepare_voc_dataset(
                        dataset_dir=config.dataset_dir,
                        output_dir=str(output_root),
                        dataset_name=config.dataset_name,
                        model_names=model_names,
                        include_unannotated=config.include_unannotated,
                        log_callback=log_callback,
                    )
            emit_progress(30)

            if is_cancelled():
                return EvalResult(success=False, message=_CANCEL_MSG)

            emit_log("[INFO] 开始验证，请稍候...")
            device = config.device if config.device not in ("auto", "") else None
            results = model.val(
                data=data_yaml,
                imgsz=int(config.imgsz),
                batch=int(config.batch),
                device=device,
                conf=float(config.conf),
                iou=float(config.iou),
                project=str(output_root),
                name=run_name,
                exist_ok=True,
                verbose=False,
                plots=True,
                save_json=True,
            )
            emit_progress(80)

            if is_cancelled():
                return EvalResult(success=False, message=_CANCEL_MSG)

            box = getattr(results, "box", None)
            map50 = _safe_float(getattr(box, "map50", None), 0.0)
            map50_95 = _safe_float(getattr(box, "map", None), 0.0)
            precision = _safe_float(getattr(box, "mp", None), 0.0)
            recall = _safe_float(getattr(box, "mr", None), 0.0)
            denom = precision + recall
            f1 = 0.0 if denom <= 0 else (2.0 * precision * recall / denom)

            per_class = _extract_per_class(results)

            metrics = EvalMetrics(
                map50=map50,
                map50_95=map50_95,
                precision=precision,
                recall=recall,
                f1=f1,
                per_class=per_class,
            )

            save_dir = str(getattr(results, "save_dir", output_root / run_name))
            artifacts = discover_yolo_plots(save_dir)
            if not artifacts:
                artifacts.update(generate_fallback_charts(metrics, save_dir))

            emit_log(f"[INFO] 验证完成，结果目录: {save_dir}")
            emit_log(
                "[METRIC] "
                f"mAP50={metrics.map50:.4f}, "
                f"mAP50-95={metrics.map50_95:.4f}, "
                f"P={metrics.precision:.4f}, "
                f"R={metrics.recall:.4f}, "
                f"F1={metrics.f1:.4f}"
            )
            emit_progress(100)

            return EvalResult(
                success=True,
                message="验证完成",
                save_dir=save_dir,
                data_yaml=data_yaml,
                metrics=metrics,
                artifacts=artifacts,
                raw_summary={"run_name": run_name},
            )
        except Exception as e:
            emit_log(f"[ERROR] 验证失败: {e}")
            return EvalResult(success=False, message=str(e))
        finally:
            del model
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def _extract_per_class(results) -> Dict[str, Dict[str, float]]:
    """Extract per-class metrics from YOLO validation results."""
    per_class: Dict[str, Dict[str, float]] = {}
    box = getattr(results, "box", None)
    if box is None:
        return per_class
    try:
        ap_cls = getattr(box, "ap_class_index", None)
        class_p = getattr(box, "p", None)
        class_r = getattr(box, "r", None)
        class_ap50 = getattr(box, "ap50", None)
        names = getattr(results, "names", {})
        if ap_cls is None or not names:
            return per_class
        for i, cls_idx in enumerate(ap_cls):
            cls_name = names.get(int(cls_idx), str(cls_idx))
            entry: Dict[str, float] = {}
            if class_p is not None and i < len(class_p):
                entry["precision"] = _safe_float(class_p[i])
            if class_r is not None and i < len(class_r):
                entry["recall"] = _safe_float(class_r[i])
            if class_ap50 is not None and i < len(class_ap50):
                entry["ap50"] = _safe_float(class_ap50[i])
            per_class[cls_name] = entry
    except Exception:
        pass
    return per_class
