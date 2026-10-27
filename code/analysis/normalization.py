"""Volume/count normalization on the selected fixed-center fitted path."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PIXEL_SIZE_MM = 0.01684976915031592
ELECTRODE_THICKNESS_MM = 0.4886433017
AXIAL_SPAN_MM = 1591 * PIXEL_SIZE_MM
BIN_ORDER = ["< 0.2", "0.2 - 0.3", ">= 0.3"]
CLASS_FILES = {
    "Internal": "Internal_object_mapping.csv",
    "Detach": "Detach_object_mapping.csv",
    "Surface": "Surface_object_mapping.csv",
}


def _build_normalization_table(
    mappings: dict[str, pd.DataFrame],
    exposure_table: pd.DataFrame,
) -> pd.DataFrame:
    exposure = exposure_table.set_index("curvature_bin")["available_path_length_mm"]
    total_path_length = float(exposure.sum())
    geometric_proxy_by_bin = exposure * ELECTRODE_THICKNESS_MM * AXIAL_SPAN_MM

    rows: list[dict[str, float | int | str]] = []
    for class_name, table in mappings.items():
        grouped = table.groupby("curvature_bin", sort=False)
        total_count = int(len(table))
        total_volume_voxel = float(table["volume_voxel"].sum())
        total_volume_mm3 = float(table["volume_mm3"].sum())
        for curvature_bin in BIN_ORDER:
            subset = grouped.get_group(curvature_bin)
            object_count = int(len(subset))
            volume_voxel = float(subset["volume_voxel"].sum())
            volume_mm3 = float(subset["volume_mm3"].sum())
            path_length_mm = float(exposure.loc[curvature_bin])
            path_share = path_length_mm / total_path_length
            volume_share = volume_mm3 / total_volume_mm3
            count_share = object_count / total_count
            proxy_volume_mm3 = float(geometric_proxy_by_bin.loc[curvature_bin])
            rows.append(
                {
                    "feature_class": class_name,
                    "curvature_bin": curvature_bin,
                    "object_count": object_count,
                    "raw_volume_voxel": volume_voxel,
                    "raw_volume_mm3": volume_mm3,
                    "available_path_length_mm": path_length_mm,
                    "path_length_share": path_share,
                    "raw_volume_share": volume_share,
                    "raw_count_share": count_share,
                    "volume_per_path_mm3_per_mm": volume_mm3 / path_length_mm,
                    "count_per_path_per_mm": object_count / path_length_mm,
                    "volume_share_to_length_share": volume_share / path_share,
                    "count_share_to_length_share": count_share / path_share,
                    "geometric_proxy_volume_mm3": proxy_volume_mm3,
                    "defect_volume_fraction_geometric_proxy": (
                        volume_mm3 / proxy_volume_mm3
                    ),
                    "geometric_proxy_is_not_measured_cathode_volume": True,
                }
            )
    return pd.DataFrame(rows)


def _make_plot(table: pd.DataFrame, output_root: Path) -> dict[str, str]:
    classes = list(CLASS_FILES)
    x = np.arange(len(BIN_ORDER), dtype=float)
    width = 0.13
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    colours = {
        "Internal": "#4C78A8",
        "Detach": "#F58518",
        "Surface": "#54A24B",
    }

    for class_index, class_name in enumerate(classes):
        subset = (
            table[table["feature_class"] == class_name]
            .set_index("curvature_bin")
            .loc[BIN_ORDER]
        )
        offset = (class_index - 1) * width
        axes[0].bar(
            x + offset,
            subset["raw_volume_share"],
            width,
            color=colours[class_name],
            alpha=0.9,
            label=class_name,
        )
        axes[1].bar(
            x + offset,
            subset["raw_count_share"],
            width,
            color=colours[class_name],
            alpha=0.9,
            label=class_name,
        )
        axes[2].plot(
            x,
            subset["volume_share_to_length_share"],
            marker="o",
            color=colours[class_name],
            label=f"{class_name} volume",
        )
        axes[2].plot(
            x,
            subset["count_share_to_length_share"],
            marker="x",
            linestyle="--",
            color=colours[class_name],
            label=f"{class_name} count",
        )

    path_share = (
        table.drop_duplicates("curvature_bin")
        .set_index("curvature_bin")
        .loc[BIN_ORDER, "path_length_share"]
    )
    axes[0].plot(x, path_share, color="black", marker="s", label="arc exposure")
    axes[1].plot(x, path_share, color="black", marker="s", label="arc exposure")
    axes[0].set_title("Raw volume share vs exposure")
    axes[1].set_title("Raw object-count share vs exposure")
    axes[2].axhline(1.0, color="black", linewidth=0.8)
    axes[2].set_title("Share-to-exposure enrichment")
    axes[2].set_ylabel("share / path-length share")
    for axis in axes[:2]:
        axis.set_ylabel("fraction")
    for axis in axes:
        axis.set_xticks(x, BIN_ORDER)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].legend(fontsize=8, ncol=2)
    axes[2].legend(fontsize=7, ncol=2)

    paths: dict[str, str] = {}
    for suffix, kwargs in (("png", {"dpi": 220}), ("svg", {})):
        path = output_root / f"defect_exposure_normalization.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths[suffix] = str(path)
    plt.close(fig)
    return paths
