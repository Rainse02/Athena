"""Calculate per-case mean and pooled metrics from nnU-Net validation counts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


from settings import DATA_ROOT, OUTPUT_ROOT, relative_metadata

SOURCE = DATA_ROOT / "validation" / "per_case_counts.json"
OUTPUT = OUTPUT_ROOT / "validation" / "summary.json"
LABELS = {"1": "Detach", "2": "Internal", "3": "Surface"}


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def valid_mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None and math.isfinite(value)]
    return sum(valid) / len(valid) if valid else None


def main() -> None:
    saved = json.loads(SOURCE.read_text(encoding="utf-8"))
    result = {
        "source": str(SOURCE),
        "case_count": len(saved["metric_per_case"]),
        "aggregation": "Per-case means and pooled TP, FP and FN counts",
        "selected_dice": "nnU-Net final validation mean of per-case Dice, ignoring NaN",
        "selected_iou": "nnU-Net final validation mean of per-case IoU, ignoring NaN",
        "complementary_precision_recall": "pooled counts across the same validation cases",
        "classes": {},
    }
    for label, name in LABELS.items():
        records = [case["metrics"][label] for case in saved["metric_per_case"]]
        totals = {
            key: sum(row[key] for row in records) for key in ("TP", "FP", "FN", "TN")
        }
        per_case_dice = []
        per_case_iou = []
        for row in records:
            dice = ratio(2 * row["TP"], 2 * row["TP"] + row["FP"] + row["FN"])
            iou = ratio(row["TP"], row["TP"] + row["FP"] + row["FN"])
            per_case_dice.append(dice)
            per_case_iou.append(iou)
            for metric, recomputed in (("Dice", dice), ("IoU", iou)):
                stored = row[metric]
                if recomputed is None:
                    assert stored is None, (name, metric, stored)
                else:
                    assert math.isclose(recomputed, stored, abs_tol=1e-12), (
                        name,
                        metric,
                    )
        mean_dice = valid_mean(per_case_dice)
        mean_iou = valid_mean(per_case_iou)
        assert math.isclose(mean_dice, saved["mean"][label]["Dice"], abs_tol=1e-12)
        assert math.isclose(mean_iou, saved["mean"][label]["IoU"], abs_tol=1e-12)
        tp, fp, fn = (totals[key] for key in ("TP", "FP", "FN"))
        result["classes"][name] = {
            "counts": totals,
            "valid_case_count": sum(value is not None for value in per_case_dice),
            "framework_mean_dice": saved["mean"][label]["Dice"],
            "recomputed_mean_dice": mean_dice,
            "framework_mean_iou": saved["mean"][label]["IoU"],
            "recomputed_mean_iou": mean_iou,
            "pooled_dice": ratio(2 * tp, 2 * tp + fp + fn),
            "pooled_iou": ratio(tp, tp + fp + fn),
            "pooled_precision": ratio(tp, tp + fp),
            "pooled_recall": ratio(tp, tp + fn),
        }
    result["all_saved_case_and_summary_scores_reproduced"] = True
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(relative_metadata(result), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "class",
        "valid_case_count",
        "Dice",
        "IoU",
        "precision",
        "recall",
        "TP",
        "FP",
        "FN",
        "TN",
    ]
    with OUTPUT.with_name("Table_S4_validation_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for name, metrics in result["classes"].items():
            writer.writerow(
                {
                    "class": name,
                    "valid_case_count": metrics["valid_case_count"],
                    "Dice": metrics["recomputed_mean_dice"],
                    "IoU": metrics["recomputed_mean_iou"],
                    "precision": metrics["pooled_precision"],
                    "recall": metrics["pooled_recall"],
                    **metrics["counts"],
                }
            )
    print("Three-class validation metrics:", OUTPUT.parent)


if __name__ == "__main__":
    main()
