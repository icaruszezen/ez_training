"""验证结果可视化。"""

from pathlib import Path
from typing import Dict

from ez_traing.evaluation.models import EvalMetrics


def discover_yolo_plots(save_dir: str) -> Dict[str, str]:
    """发现 YOLO 自动生成的图表文件。"""
    root = Path(save_dir)
    if not root.exists():
        return {}

    candidates = {
        "confusion_matrix": ["confusion_matrix.png", "confusion_matrix_normalized.png"],
        "pr_curve": ["PR_curve.png"],
        "f1_curve": ["F1_curve.png"],
        "p_curve": ["P_curve.png"],
        "r_curve": ["R_curve.png"],
        "results": ["results.png"],
    }

    found: Dict[str, str] = {}
    for key, names in candidates.items():
        for name in names:
            path = root / name
            if path.exists():
                found[key] = str(path)
                break

    for pred_img in sorted(root.glob("val_batch*_pred.png")):
        found[pred_img.stem] = str(pred_img)

    return found


def generate_fallback_charts(metrics: EvalMetrics, output_dir: str) -> Dict[str, str]:
    """当 YOLO 未生成图表时，回退生成基础柱状图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        chart_path = root / "metrics_overview.png"

        labels = ["mAP50", "mAP50-95", "Precision", "Recall", "F1"]
        values = [
            float(metrics.map50),
            float(metrics.map50_95),
            float(metrics.precision),
            float(metrics.recall),
            float(metrics.f1),
        ]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(labels, values)
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Validation Metrics")
        ax.set_ylabel("Score")

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.02,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        fig.tight_layout()
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        return {"metrics_overview": str(chart_path)}
    except Exception:
        return {}
