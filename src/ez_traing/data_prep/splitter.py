"""数据集划分工具。"""

import random
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List, Optional, Tuple

from ez_traing.data_prep.models import DatasetSample


def _normalize_base_stem(stem: str) -> str:
    """提取同源样本主干名，降低增强/拷贝变体被分到不同集合的风险。"""
    value = stem.lower().strip()

    # 常见增强后缀：xxx_aug1 / xxx_flip / xxx_rot90 / xxx_noise2 ...
    suffix_patterns = [
        r"([_\-](aug|copy|dup|flip|flipped|mirror|rot|rotate|noise|blur|crop|resize|color|bright|contrast|gamma|hsv|clahe|persp|affine|shift|trans)\d*)+$",
        r"([_\-](v|ver|version)\d+)+$",
    ]
    for pattern in suffix_patterns:
        value = re.sub(pattern, "", value)

    value = re.sub(r"[^\w]+", "_", value).strip("_")
    return value or stem.lower()


def _leakage_group_key(sample: DatasetSample, dataset_root: Optional[Path] = None) -> str:
    """生成防泄露分组 key：同目录同源主干视为一组。

    使用相对于 *dataset_root* 的完整父路径作为目录标识，
    避免不同子树下同名目录的样本被错误归组。
    """
    path: Path = sample.image_path
    if dataset_root is not None:
        try:
            parent = path.relative_to(dataset_root).parent.as_posix().lower()
        except ValueError:
            parent = path.parent.name.lower()
    else:
        parent = path.parent.name.lower()
    base = _normalize_base_stem(path.stem)
    return f"{parent}::{base}"


def split_train_val(
    samples: List[DatasetSample],
    train_ratio: float,
    seed: int,
    dataset_root: Optional[Path] = None,
) -> Tuple[List[DatasetSample], List[DatasetSample]]:
    """按同源分组划分 train/val，避免数据泄露。"""
    if not samples:
        return [], []

    groups: DefaultDict[str, List[DatasetSample]] = defaultdict(list)
    for sample in samples:
        groups[_leakage_group_key(sample, dataset_root)].append(sample)

    grouped_items = list(groups.items())
    if len(grouped_items) < 2:
        raise ValueError(
            "可用于无泄露划分的同源分组不足 2 组，请增加数据或调整样本命名后重试。"
        )

    rnd = random.Random(seed)
    rnd.shuffle(grouped_items)

    total_samples = len(samples)
    target_train = max(1, min(int(total_samples * train_ratio), total_samples - 1))

    train_samples: List[DatasetSample] = []
    val_samples: List[DatasetSample] = []

    for _, group in grouped_items:
        # 按组装配 train，防止同组样本进入 val 造成泄露。
        if not train_samples or len(train_samples) + len(group) <= target_train:
            train_samples.extend(group)
        else:
            val_samples.extend(group)

    # 兜底保证两个集合都不为空（仍保持按组移动）
    if not val_samples:
        last_group = grouped_items[-1][1]
        moved = set(id(item) for item in last_group)
        train_samples = [s for s in train_samples if id(s) not in moved]
        val_samples = list(last_group)

    if not train_samples:
        first_group = grouped_items[0][1]
        moved = set(id(item) for item in first_group)
        val_samples = [s for s in val_samples if id(s) not in moved]
        train_samples = list(first_group)

    return train_samples, val_samples
