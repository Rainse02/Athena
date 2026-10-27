"""Recompute axial node lag statistics from the recorded 1590-row segmented profiles."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from settings import DATA_ROOT, OUTPUT_ROOT, relative_metadata


SOURCE = DATA_ROOT / "axial" / "axial_profiles.csv"
DEFAULT_OUTPUT = OUTPUT_ROOT / "nodes"
EXPECTED_COMMON_ROWS = 1590
MARKER_THRESHOLD = 128
AUTOCORR_LAG_MIN = 50
AUTOCORR_LAG_MAX = 100
DETREND_WINDOW = 151
PEAK_SMOOTHING_SIGMA = 3.0
PEAK_DISTANCE = 50
PEAK_PROMINENCE_FRACTION = 0.05


def _json_safe(value: Any) -> Any:
    """Convert arrays, paths and non-finite floats to JSON values."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    """Pearson correlation after removing non-finite pairs."""

    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    finite = np.isfinite(first_array) & np.isfinite(second_array)
    first_array = first_array[finite]
    second_array = second_array[finite]
    if first_array.size < 2:
        return float("nan")
    first_centered = first_array - np.mean(first_array)
    second_centered = second_array - np.mean(second_array)
    denominator = np.sqrt(np.sum(first_centered**2) * np.sum(second_centered**2))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(first_centered * second_centered) / denominator)


def _lag_pairs(
    first: np.ndarray,
    second: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nonwrapping pairs for corr(first[z], second[z+lag])."""

    if lag > 0:
        return first[:-lag], second[lag:]
    if lag < 0:
        return first[-lag:], second[:lag]
    return first, second


def _lag_correlations(
    first: np.ndarray,
    second: np.ndarray,
    lags: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return Pearson values and finite overlap counts for nonwrapping lags."""

    correlations: list[float] = []
    overlaps: list[int] = []
    for lag in lags:
        first_pair, second_pair = _lag_pairs(first, second, int(lag))
        correlations.append(_pearson(first_pair, second_pair))
        finite = np.isfinite(first_pair) & np.isfinite(second_pair)
        overlaps.append(int(np.count_nonzero(finite)))
    return np.asarray(correlations, dtype=float), np.asarray(overlaps, dtype=int)


def _best_lag(lags: np.ndarray, scores: np.ndarray) -> tuple[int, float]:
    """Choose the highest score, preferring the smallest absolute lag on ties."""

    finite = np.isfinite(scores)
    if not np.any(finite):
        return 0, float("nan")
    maximum = float(np.nanmax(scores))
    candidates = lags[finite & np.isclose(scores, maximum, rtol=0.0, atol=1e-12)]
    chosen = sorted((int(lag) for lag in candidates), key=lambda lag: (abs(lag), lag))[
        0
    ]
    return chosen, maximum


def _reflected_running_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving mean using reflected samples at both array ends."""

    if window <= 0 or window % 2 != 1:
        raise ValueError(
            "The reflected running-mean window must be a positive odd integer"
        )
    array = np.asarray(values, dtype=float)
    half_window = window // 2
    padded = np.pad(array, (half_window, half_window), mode="reflect")
    kernel = np.full(window, 1.0 / window, dtype=float)
    return np.convolve(padded, kernel, mode="valid")


def _node_autocorrelation(profile: np.ndarray) -> pd.DataFrame:
    """Compute global-mean-removed, nonwrapping node autocorrelation."""

    values = np.asarray(profile, dtype=float)
    centered = values - np.mean(values)
    denominator = float(np.sum(centered**2))
    rows: list[dict[str, Any]] = []
    for lag in range(0, AUTOCORR_LAG_MAX + 1):
        first = centered[: len(centered) - lag] if lag else centered
        second = centered[lag:] if lag else centered
        numerator = float(np.sum(first * second))
        value = numerator / denominator if denominator > 0 else float("nan")
        rows.append(
            {
                "lag_pixels": int(lag),
                "autocorrelation": value,
                "mean_removed": True,
                "normalization": "global_centered_energy_at_lag_zero",
                "within_50_100_search": bool(
                    AUTOCORR_LAG_MIN <= lag <= AUTOCORR_LAG_MAX
                ),
            }
        )
    return pd.DataFrame(rows)


def _period_estimates(profile: np.ndarray) -> dict[str, Any]:
    """Estimate the node period from autocorrelation, FFT, and optional peaks."""

    values = np.asarray(profile, dtype=float)
    autocorrelation = _node_autocorrelation(values)
    restricted = autocorrelation.loc[autocorrelation["within_50_100_search"]]
    autocorr_index = int(restricted["autocorrelation"].idxmax())
    autocorr_peak_lag = int(autocorrelation.loc[autocorr_index, "lag_pixels"])
    autocorr_peak_value = float(autocorrelation.loc[autocorr_index, "autocorrelation"])

    centered = values - np.mean(values)
    frequencies = np.fft.rfftfreq(len(centered), d=1.0)
    power = np.abs(np.fft.rfft(centered)) ** 2
    spectral_mask = (
        (frequencies > 0)
        & (frequencies >= 1.0 / AUTOCORR_LAG_MAX)
        & (frequencies <= 1.0 / AUTOCORR_LAG_MIN)
    )
    spectral_indices = np.flatnonzero(spectral_mask)
    if spectral_indices.size == 0:
        spectral_period = float("nan")
        spectral_frequency = float("nan")
        spectral_power = float("nan")
    else:
        spectral_index = int(spectral_indices[np.argmax(power[spectral_indices])])
        spectral_frequency = float(frequencies[spectral_index])
        spectral_period = float(1.0 / spectral_frequency)
        spectral_power = float(power[spectral_index])

    smoothed = gaussian_filter1d(
        values,
        sigma=PEAK_SMOOTHING_SIGMA,
        mode="reflect",
    )
    prominence = PEAK_PROMINENCE_FRACTION * float(np.ptp(values))
    if prominence > 0:
        peak_rows, peak_properties = find_peaks(
            smoothed,
            distance=PEAK_DISTANCE,
            prominence=prominence,
        )
        peak_prominences = peak_properties.get("prominences", np.array([]))
    else:
        peak_rows = np.array([], dtype=int)
        peak_prominences = np.array([], dtype=float)
    peak_spacing = np.diff(peak_rows).astype(float)

    if not math.isfinite(spectral_period):
        raise ValueError("Unable to derive a node period in the 50-100 pixel band")
    half_period_limit = int(math.floor(spectral_period / 2.0))
    return {
        "autocorrelation_peak_lag_pixels": autocorr_peak_lag,
        "autocorrelation_peak_value": autocorr_peak_value,
        "spectral_dominant_frequency_cycles_per_pixel": spectral_frequency,
        "spectral_dominant_period_pixels": spectral_period,
        "spectral_dominant_power": spectral_power,
        "period_used_for_lag_search_pixels": spectral_period,
        "half_period_lag_limit_pixels": half_period_limit,
        "row_peak_method": {
            "smoothing": "reflected centered Gaussian",
            "sigma_pixels": PEAK_SMOOTHING_SIGMA,
            "distance_pixels": PEAK_DISTANCE,
            "prominence_fraction_of_raw_profile_ptp": PEAK_PROMINENCE_FRACTION,
            "prominence_threshold": prominence,
            "peak_count": int(len(peak_rows)),
            "peak_rows_pixels": [int(value) for value in peak_rows],
            "spacing_pixels": [float(value) for value in peak_spacing],
            "median_spacing_pixels": (
                float(np.median(peak_spacing)) if peak_spacing.size else float("nan")
            ),
            "mean_spacing_pixels": (
                float(np.mean(peak_spacing)) if peak_spacing.size else float("nan")
            ),
            "peak_prominences": [float(value) for value in peak_prominences],
        },
    }


def _phase_value(lag: int, period: float) -> tuple[float, float]:
    fraction = float(lag / period) if period and math.isfinite(period) else float("nan")
    return fraction, (
        float(360.0 * fraction) if math.isfinite(fraction) else float("nan")
    )


def _lag_scan_table(
    defect_profiles: dict[str, np.ndarray],
    node_profile: np.ndarray,
    period_pixels: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Scan raw and detrended Detachment profiles within the derived half-period."""

    half_period_limit = int(math.floor(period_pixels / 2.0))
    lags = np.arange(-half_period_limit, half_period_limit + 1, dtype=int)
    node_detrended = node_profile - _reflected_running_mean(
        node_profile, DETREND_WINDOW
    )
    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for defect_name, defect_profile in defect_profiles.items():
        defect_detrended = defect_profile - _reflected_running_mean(
            defect_profile,
            DETREND_WINDOW,
        )
        for variant_name, defect_values, node_values in (
            ("raw", defect_profile, node_profile),
            ("detrended_reflected151", defect_detrended, node_detrended),
        ):
            correlations, overlaps = _lag_correlations(defect_values, node_values, lags)
            half_mask = np.abs(lags) <= half_period_limit
            half_lag, half_score = _best_lag(lags[half_mask], correlations[half_mask])
            zero_index = int(np.flatnonzero(lags == 0)[0])
            plus12_index = int(np.flatnonzero(lags == 12)[0])
            variant_summary = {
                "zero_lag": {
                    "lag_pixels": 0,
                    "pearson_r": float(correlations[zero_index]),
                    "lag_fraction_of_period": 0.0,
                    "phase_degrees": 0.0,
                },
                "plus12_lag": {
                    "lag_pixels": 12,
                    "pearson_r": float(correlations[plus12_index]),
                    "lag_fraction_of_period": _phase_value(12, period_pixels)[0],
                    "phase_degrees": _phase_value(12, period_pixels)[1],
                },
                "best_within_half_period": {
                    "lag_pixels": half_lag,
                    "pearson_r": half_score,
                    "lag_fraction_of_period": _phase_value(half_lag, period_pixels)[0],
                    "phase_degrees": _phase_value(half_lag, period_pixels)[1],
                },
            }
            summary.setdefault(defect_name, {})[variant_name] = variant_summary
            for lag, correlation, overlap in zip(lags, correlations, overlaps):
                fraction, degrees = _phase_value(int(lag), period_pixels)
                all_rows.append(
                    {
                        "defect_profile": defect_name,
                        "node_profile": "node_nonzero_row_fraction",
                        "profile_variant": variant_name,
                        "lag_pixels": int(lag),
                        "lag_fraction_of_period": fraction,
                        "phase_degrees": degrees,
                        "pearson_r": float(correlation),
                        "overlap_samples": int(overlap),
                        "within_derived_half_period": bool(
                            abs(lag) <= half_period_limit
                        ),
                        "lag_scan_limit_pixels": half_period_limit,
                        "derived_half_period_limit_pixels": half_period_limit,
                        "is_zero_lag": bool(lag == 0),
                        "is_plus12_lag": bool(lag == 12),
                        "is_best_within_derived_half_period": bool(lag == half_lag),
                    }
                )
    return pd.DataFrame(all_rows), summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(relative_metadata(_json_safe(payload)), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def run(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Calculate node periods and lag correlations from common axial profiles."""
    table = pd.read_csv(SOURCE)
    if len(table) != EXPECTED_COMMON_ROWS:
        raise ValueError("The recorded comparison requires exactly 1590 common rows.")
    node_profile = table["node_nonzero_row_fraction"].to_numpy(dtype=float)
    defect_profiles = {
        "Detach": table["Detach_fullheight_defect_fraction"].to_numpy(dtype=float),
    }
    periods = _period_estimates(node_profile)
    lag_table, lag_summary = _lag_scan_table(
        defect_profiles,
        node_profile,
        float(periods["period_used_for_lag_search_pixels"]),
    )
    autocorrelation = _node_autocorrelation(node_profile)
    output_directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_directory / "axial_profiles.csv", index=False)
    lag_table.to_csv(output_directory / "lag_scan.csv", index=False)
    autocorrelation.to_csv(output_directory / "node_autocorrelation.csv", index=False)
    summary = {
        "analysis": "axial_node_alignment",
        "source": str(SOURCE),
        "input_level": "Recorded segmentation-derived defect fractions and processed node-marker row profiles",
        "common_rows": len(table),
        "classes": ["Detach"],
        "period_estimates": periods,
        "lag_definition": "corr(D[z], node[z+lag]) with nonwrapping overlap",
        "lag_search_results": lag_summary,
        "reported_lags_pixels": [0, 12],
        "scan_range_pixels": [
            -int(periods["half_period_lag_limit_pixels"]),
            int(periods["half_period_lag_limit_pixels"]),
        ],
        "pairing": "Acquisition order over the first 1590 common axial rows",
    }
    _write_json(output_directory / "summary.json", summary)
    print("Node alignment:", output_directory)
    return summary


if __name__ == "__main__":
    run()
