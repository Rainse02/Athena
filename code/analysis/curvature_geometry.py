"""Finite-thickness geometry helpers used by the selected fixed-center analysis."""

from __future__ import annotations

import numpy as np


def _as_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def spiral_func(theta: np.ndarray | float, a: float, b: float) -> np.ndarray | float:
    return np.asarray(a) + np.asarray(b) * np.asarray(theta)


def parallel_curve_curvature_interval(
    curvature_1_per_mm: np.ndarray | float,
    thickness_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return curvature bounds for parallel curves at half-thickness offsets.

    For a centerline curvature ``k`` and total layer thickness ``h``, the
    requested conservative bounds are ``k / (1 + k*h/2)`` and
    ``k / (1 - k*h/2)``.  The interval is marked invalid when either
    denominator is non-positive or when the inputs are non-finite.  The
    function is vectorized and preserves the shape of the curvature input.
    """

    curvature = np.asarray(curvature_1_per_mm, dtype=float)
    thickness = _as_float(thickness_mm, float("nan"))
    lower = np.full(curvature.shape, np.nan, dtype=float)
    upper = np.full(curvature.shape, np.nan, dtype=float)
    if not np.isfinite(thickness) or thickness < 0.0:
        return lower, upper, np.zeros(curvature.shape, dtype=bool)

    half_thickness = 0.5 * thickness
    lower_denominator = 1.0 + curvature * half_thickness
    upper_denominator = 1.0 - curvature * half_thickness
    valid = (
        np.isfinite(curvature)
        & np.isfinite(lower_denominator)
        & np.isfinite(upper_denominator)
        & (lower_denominator > 0.0)
        & (upper_denominator > 0.0)
    )
    np.divide(
        curvature,
        lower_denominator,
        out=lower,
        where=valid,
    )
    np.divide(
        curvature,
        upper_denominator,
        out=upper,
        where=valid,
    )
    return lower, upper, valid


def archimedean_adjacent_turn_normal_separation(
    theta: np.ndarray | float,
    a_pixel: float,
    b_pixel: float,
) -> np.ndarray:
    """Estimate normal separation between adjacent Archimedean turns.

    At a given polar angle, adjacent turns differ radially by ``2*pi*b``.
    Projecting that radial difference onto the local normal gives
    ``abs(2*pi*b) * r / sqrt(r**2 + b**2)``.  This is a local same-angle
    normal projection, used as an estimate rather than an exact closest-point
    distance between the two curves.
    """

    theta_array = np.asarray(theta, dtype=float)
    radius = np.asarray(spiral_func(theta_array, a_pixel, b_pixel), dtype=float)
    denominator = np.hypot(radius, float(b_pixel))
    separation = np.full(radius.shape, np.nan, dtype=float)
    valid = np.isfinite(radius) & (radius > 0.0) & np.isfinite(denominator)
    pitch = abs(2.0 * np.pi * float(b_pixel))
    np.divide(
        pitch * np.abs(radius),
        denominator,
        out=separation,
        where=valid,
    )
    return separation
