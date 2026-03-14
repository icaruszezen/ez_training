"""验证报告导出。"""

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from ez_traing.evaluation.models import EvalConfig, EvalResult


def _round_floats(obj: Any, precision: int = 6) -> Any:
    """Recursively round floats to consistent precision for JSON output."""
    if isinstance(obj, float):
        return 0.0 if not math.isfinite(obj) else round(obj, precision)
    if isinstance(obj, dict):
        return {k: _round_floats(v, precision) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(item, precision) for item in obj]
    return obj


def export_reports(result: EvalResult, config: EvalConfig, output_dir: str) -> Dict[str, str]:
    """导出验证报告到指定目录。"""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    metrics_json = root / "metrics.json"
    metrics_csv = root / "metrics.csv"

    if metrics_json.exists() or metrics_csv.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_json = root / f"metrics_{ts}.json"
        metrics_csv = root / f"metrics_{ts}.csv"

    payload = _round_floats({
        "config": config.to_dict(),
        "result": result.to_dict(),
    })
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(metrics_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        if result.metrics:
            writer.writerow(["mAP50", f"{result.metrics.map50:.6f}"])
            writer.writerow(["mAP50-95", f"{result.metrics.map50_95:.6f}"])
            writer.writerow(["Precision", f"{result.metrics.precision:.6f}"])
            writer.writerow(["Recall", f"{result.metrics.recall:.6f}"])
            writer.writerow(["F1", f"{result.metrics.f1:.6f}"])
        writer.writerow(["save_dir", result.save_dir])
        writer.writerow(["message", result.message])

    return {
        "metrics_json": str(metrics_json),
        "metrics_csv": str(metrics_csv),
    }
