"""Summarize the broad Detachment profile without changing node alignment.

This is a descriptive equal-height summary of the existing 1,590-row dataset.
It does not estimate independent defect-initiation rates or fit a damage model.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


from package_paths import DATA_ROOT, OUTPUT_ROOT, relative_metadata, run_cli

SOURCE = DATA_ROOT / "axial" / "axial_profiles.csv"
OUTPUT = OUTPUT_ROOT / "axial_ends"
DEFECT_COLUMN = "Detach_fullheight_defect_fraction"
NODE_COLUMN = "node_nonzero_row_fraction"


def summarize(rows: list[dict[str, str]], label: str) -> dict[str, object]:
    """Average the stored fractions over one explicitly specified interval."""
    return {
        "region": label,
        "first_axial_index": int(rows[0]["axial_index"]),
        "last_axial_index": int(rows[-1]["axial_index"]),
        "row_count": len(rows),
        "mean_detach_area_fraction": mean(float(row[DEFECT_COLUMN]) for row in rows),
        "mean_node_marker_fraction": mean(float(row[NODE_COLUMN]) for row in rows),
    }


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    with SOURCE.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1590:
        raise ValueError(
            "This summary is defined for the existing 1,590-row comparison."
        )

    bin_size = len(rows) // 10
    bins = [
        summarize(
            rows[index * bin_size : (index + 1) * bin_size],
            f"height_bin_{index + 1}",
        )
        for index in range(10)
    ]
    regions = [
        summarize(rows[:bin_size], "lower_10_percent"),
        summarize(rows[3 * bin_size : 7 * bin_size], "central_40_percent"),
        summarize(rows[-bin_size:], "upper_10_percent"),
    ]
    central_mean = regions[1]["mean_detach_area_fraction"]
    for region in regions:
        region["detach_mean_relative_to_central"] = (
            region["mean_detach_area_fraction"] / central_mean
        )

    summary = {
        "source": str(SOURCE),
        "common_rows": len(rows),
        "coordinate": "Increasing slice index is upward under the paper's Y=w convention.",
        "measurement": "Mean of stored segmented Detachment area fractions per axial slice.",
        "interval_definition": "Ten equal bins of 159 rows; central comparison uses bins 4-7.",
        "scope": (
            "Intervals refer to the common analyzed height, not independently segmented "
            "distances from physical electrode cut edges. Fractions retain the original "
            "CT analysis exposure denominator; they are not electrode-material loss rates."
        ),
        "interpretation": (
            "Descriptive broad-profile comparison; no significance test, specimen-specific "
            "boundary calibration, node-level risk estimate, or new segmentation."
        ),
        "bins": bins,
        "regions": regions,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_table(OUTPUT / "height_bins.csv", bins)
    write_table(OUTPUT / "end_central_comparison.csv", regions)
    (OUTPUT / "summary.json").write_text(
        json.dumps(relative_metadata(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(regions, indent=2))


if __name__ == "__main__":
    run_cli(main, __doc__)
