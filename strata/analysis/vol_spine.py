"""Forward volatility spine construction and stress diagnostics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ForwardVolPoint:
    """Single forward-vol node along the term-structure spine."""

    expiry: datetime
    atm_strike: float
    atm_iv: float
    surface_iv: float
    residual: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-friendly dict."""
        return {
            "expiry": self.expiry.isoformat(),
            "atm_strike": self.atm_strike,
            "atm_iv": self.atm_iv,
            "surface_iv": self.surface_iv,
            "residual": self.residual,
        }


def build_forward_vol_spine(
    term_structure: Sequence[Dict[str, Any]],
    now: Optional[datetime] = None,
) -> List[ForwardVolPoint]:
    """
    Construct the forward volatility spine and residuals.

    Args:
        term_structure: Iterable of points with keys:
            - expiry: datetime or ISO8601 string
            - atm_iv: ATM implied vol for expiry
            - atm_strike: ATM strike for expiry
            - surface_iv: Optional smooth IV surface estimate at ATM
        now: Reference time for tenor calculation (defaults to utcnow)

    Returns:
        Ordered list of ForwardVolPoint (front to back)
    """
    if now is None:
        now = datetime.utcnow()

    if not term_structure:
        return []

    normalized_points: List[Dict[str, Any]] = []
    for raw in term_structure:
        expiry_raw = raw.get("expiry")
        if expiry_raw is None:
            logger.debug("Skipping term-structure point with missing expiry: %s", raw)
            continue

        expiry = pd.to_datetime(expiry_raw).to_pydatetime()
        atm_iv = float(raw.get("atm_iv", raw.get("iv", 0.0)))
        atm_strike = float(raw.get("atm_strike", raw.get("strike", 0.0)))
        surface_iv = raw.get("surface_iv")
        normalized_points.append(
            {
                "expiry": expiry,
                "tenor_days": max(
                    (expiry - now).total_seconds() / 86400.0,
                    0.01,
                ),
                "atm_iv": atm_iv,
                "atm_strike": atm_strike,
                "surface_iv": float(surface_iv) if surface_iv is not None else None,
            }
        )

    if not normalized_points:
        return []

    # Sort by expiry (front to back)
    normalized_points.sort(key=lambda x: x["expiry"])

    tenors = np.array([p["tenor_days"] for p in normalized_points], dtype=float)
    atm_ivs = np.array([p["atm_iv"] for p in normalized_points], dtype=float)
    surface_ivs = np.array(
        [
            p["surface_iv"] if p["surface_iv"] is not None else np.nan
            for p in normalized_points
        ],
        dtype=float,
    )

    # Estimate surface IV if missing using a smooth polynomial fit over tenor
    if np.isnan(surface_ivs).any():
        degree = 2 if len(tenors) >= 3 else 1
        try:
            coeffs = np.polyfit(tenors, atm_ivs, deg=degree)
            fitted_surface = np.polyval(coeffs, tenors)
            surface_ivs = np.where(np.isnan(surface_ivs), fitted_surface, surface_ivs)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Surface fit failed (%s); falling back to ATM IVs", exc)
            surface_ivs = atm_ivs.copy()

    spine: List[ForwardVolPoint] = []
    for point, surface_iv in zip(normalized_points, surface_ivs):
        residual = float(point["atm_iv"] - surface_iv)
        spine.append(
            ForwardVolPoint(
                expiry=point["expiry"],
                atm_strike=point["atm_strike"],
                atm_iv=float(point["atm_iv"]),
                surface_iv=float(surface_iv),
                residual=residual,
            )
        )

    return spine


def serialize_spine(spine: Iterable[ForwardVolPoint]) -> List[Dict[str, Any]]:
    """Convert spine points into JSON-friendly dicts."""
    return [p.to_dict() for p in spine]


def extract_residual_field(spine: Iterable[ForwardVolPoint]) -> List[Dict[str, Any]]:
    """
    Build the spine residual field SRF(expiry) = ATM_IV - Surface_IV.

    Returns:
        List of {expiry, residual, atm_iv, surface_iv}
    """
    field: List[Dict[str, Any]] = []
    for point in spine:
        field.append(
            {
                "expiry": point.expiry.isoformat(),
                "residual": point.residual,
                "atm_iv": point.atm_iv,
                "surface_iv": point.surface_iv,
            }
        )
    return field


def analyze_spine_field(
    spine_field: Sequence[Dict[str, Any]] | Sequence[ForwardVolPoint],
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Derive gradient/curvature diagnostics for the spine residual field.

    Args:
        spine_field: Residual field (dicts or ForwardVolPoint) sorted front→back
        now: Reference time for tenor calculation (defaults to utcnow)

    Returns:
        Dict with gradients, curvature, maxima, and qualitative shape label.
    """
    if now is None:
        now = datetime.utcnow()

    if not spine_field:
        return {
            "residuals": [],
            "gradient": [],
            "curvature": [],
            "front_residual": 0.0,
            "max_abs_residual": 0.0,
            "max_abs_curvature": 0.0,
            "shape": "flat",
        }

    def _as_point(item: Dict[str, Any] | ForwardVolPoint) -> ForwardVolPoint:
        if isinstance(item, ForwardVolPoint):
            return item
        expiry = pd.to_datetime(item["expiry"]).to_pydatetime()
        return ForwardVolPoint(
            expiry=expiry,
            atm_strike=float(item.get("atm_strike", 0.0)),
            atm_iv=float(item.get("atm_iv", 0.0)),
            surface_iv=float(item.get("surface_iv", 0.0)),
            residual=float(item.get("residual", 0.0)),
        )

    points = [_as_point(p) for p in spine_field]
    points.sort(key=lambda p: p.expiry)

    tenors = np.array(
        [
            max((p.expiry - now).total_seconds() / 86400.0, 0.01)
            for p in points
        ],
        dtype=float,
    )
    residuals = np.array([p.residual for p in points], dtype=float)

    gradient = np.gradient(residuals, tenors) if len(points) >= 2 else np.array([])
    curvature = (
        np.gradient(gradient, tenors) if len(points) >= 3 else np.array([])
    )

    max_abs_residual = float(np.max(np.abs(residuals))) if len(residuals) else 0.0
    max_abs_curvature = (
        float(np.max(np.abs(curvature))) if len(curvature) else 0.0
    )

    shape = classify_spine_shape(residuals, gradient)

    return {
        "residuals": [
            {"expiry": p.expiry.isoformat(), "residual": p.residual}
            for p in points
        ],
        "gradient": gradient.tolist() if gradient.size else [],
        "curvature": curvature.tolist() if curvature.size else [],
        "front_residual": float(residuals[0]) if len(residuals) else 0.0,
        "max_abs_residual": max_abs_residual,
        "max_abs_curvature": max_abs_curvature,
        "shape": shape,
    }


def classify_spine_shape(residuals: np.ndarray, gradient: np.ndarray) -> str:
    """
    Qualitative shape classification for SRF.

    Returns one of: flat, upward_sloping, inverted, humped, oscillating.
    """
    if residuals.size < 2:
        return "flat"

    mean_slope = float(np.mean(gradient)) if gradient.size else 0.0
    slope_threshold = 0.02

    if mean_slope > slope_threshold:
        return "upward_sloping"
    if mean_slope < -slope_threshold:
        return "inverted"

    # Hump if peak is interior and clearly above edges
    peak_idx = int(np.argmax(residuals))
    edge_mean = float(np.mean([residuals[0], residuals[-1]]))
    peak = float(residuals[peak_idx])
    if 0 < peak_idx < residuals.size - 1 and peak - edge_mean > 0.03:
        return "humped"

    # Oscillating if residuals change sign multiple times
    sign_changes = np.diff(np.sign(residuals))
    if np.count_nonzero(sign_changes) >= 2:
        return "oscillating"

    return "flat"
