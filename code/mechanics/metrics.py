"""Original geometric masks and percentile algorithms for final mechanics exports."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def distance_to_interface(
    x_mm: np.ndarray | float,
    y_mm: np.ndarray | float,
    vertices_mm: Iterable[Iterable[float]],
) -> np.ndarray:
    """Return point-to-polygon-edge distances in mm.

    The two vertical end segments at x = +/- L/2 are omitted. This matches the
    sampling rule used for the interface exports and prevents the artificial
    end cuts from entering the local-neighborhood comparison.
    """

    x_array, y_array = np.broadcast_arrays(
        np.asarray(x_mm, dtype=float),
        np.asarray(y_mm, dtype=float),
    )
    points = np.column_stack((x_array.ravel(), y_array.ravel()))
    vertices = np.asarray(list(vertices_mm), dtype=float)
    half_length = float(np.max(np.abs(vertices[:, 0])))
    distances = np.full(points.shape[0], np.inf, dtype=float)
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
        if (
            abs(start[0] - end[0]) <= 1e-12
            and abs(abs(start[0]) - half_length) <= 1e-12
        ):
            continue
        segment = end - start
        length_squared = float(np.dot(segment, segment))
        if length_squared == 0.0:
            continue
        projection = (points - start) @ segment / length_squared
        projection = np.clip(projection, 0.0, 1.0)
        closest = start + projection[:, None] * segment
        distances = np.minimum(distances, np.linalg.norm(points - closest, axis=1))
    return distances.reshape(x_array.shape)


def weighted_percentile(
    values: np.ndarray,
    weights: np.ndarray | None = None,
    percentile: float = 95.0,
) -> float:
    """Return a weighted nearest-rank percentile after dropping invalid rows."""

    values_array = np.asarray(values, dtype=float).ravel()
    if weights is None:
        weights_array = np.ones_like(values_array)
    else:
        weights_array = np.asarray(weights, dtype=float).ravel()
    valid = np.isfinite(values_array) & np.isfinite(weights_array) & (weights_array > 0)
    values_array = values_array[valid]
    weights_array = weights_array[valid]
    if values_array.size == 0:
        return float("nan")
    order = np.argsort(values_array)
    sorted_values = values_array[order]
    sorted_weights = weights_array[order]
    target = percentile / 100.0 * float(np.sum(sorted_weights))
    index = int(np.searchsorted(np.cumsum(sorted_weights), target, side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def interval_mask(x_mm: np.ndarray, intervals: Iterable[Iterable[float]]) -> np.ndarray:
    mask = np.zeros(x_mm.shape, dtype=bool)
    for lower, upper in intervals:
        mask |= (x_mm >= lower) & (x_mm <= upper)
    return mask


def finite_percentile_record(
    values: np.ndarray,
    weights: np.ndarray | None,
    mask: np.ndarray,
) -> dict[str, float | int]:
    selected_values = np.asarray(values, dtype=float)[mask]
    selected_weights = (
        None if weights is None else np.asarray(weights, dtype=float)[mask]
    )
    valid = np.isfinite(selected_values)
    if selected_weights is not None:
        valid &= np.isfinite(selected_weights) & (selected_weights > 0)
    selected_values = selected_values[valid]
    if selected_weights is not None:
        selected_weights = selected_weights[valid]
    p95 = weighted_percentile(selected_values, selected_weights)
    weight_sum = float(
        selected_values.size if selected_weights is None else np.sum(selected_weights)
    )
    return {
        "p95": p95,
        "sample_count": int(selected_values.size),
        "weight_sum": weight_sum,
    }


def weighted(
    values: np.ndarray, weights: np.ndarray, mask: np.ndarray, p: float = 95
) -> float:
    return weighted_percentile(values[mask], weights[mask], p)


def interface_metrics(data: np.ndarray, config: dict) -> dict:
    weight = data[:, 0]
    x_mm = data[:, 2] * 1e3
    normal = data[:, 9] / data[:, 13]
    shear = data[:, 10] / data[:, 13]
    normal_al = data[:, 11] / data[:, 13]
    shear_al = data[:, 12] / data[:, 13]
    node = interval_mask(x_mm, config["node_x_intervals_mm"])
    regions = {
        "node": node,
        "ordinary": interval_mask(x_mm, config["ordinary_x_intervals_mm"]),
        "node_lower_neck": node
        & (data[:, 1] == -1)
        & (data[:, 3] * 1e3 <= 0.05 * config["domain_mm"][1]),
    }
    result = {}
    for region, mask in regions.items():
        result[region] = {
            "normal_tensile_P95": weighted(np.maximum(normal, 0), weight, mask),
            "normal_compression_P95": weighted(np.maximum(-normal, 0), weight, mask),
            "shear_P95": weighted(np.abs(shear), weight, mask),
            "normal_tension_arc_fraction": float(
                np.sum(weight[mask & (normal > 1e-6)]) / np.sum(weight[mask])
            ),
            "al_side_normal_tensile_P95": weighted(
                np.maximum(normal_al, 0), weight, mask
            ),
            "al_side_shear_P95": weighted(np.abs(shear_al), weight, mask),
        }
    return result
