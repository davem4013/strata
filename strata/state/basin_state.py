"""Immutable basin geometry snapshot for residual clustering."""
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ResidualBasin:
    """
    First-class basin geometry representation.

    This object is intentionally minimal: center + geometry and stability hints.
    It stays JSON/DB friendly via ``to_dict`` so callers can persist legacy shapes.
    """

    center: float
    width: float
    boundary_upper: float
    boundary_lower: float
    curvature: float
    sample_count: int
    variance: float

    # Optional 2D metadata (residual_coordinate, response) for domain_dim == 2.
    domain_dim: int = 1
    center_vector: Optional[List[float]] = None  # [residual_center, response_center]
    width_vector: Optional[List[float]] = None   # axis-aligned spreads (residual, response)
    covariance: Optional[List[List[float]]] = None  # axis-aligned covariance matrix (diagonal)

    @property
    def radius(self) -> float:
        return self.width / 2.0

    def distance(self, x: float) -> float:
        """Absolute distance from x to basin center."""
        return abs(float(x) - self.center)

    def normalized_distance(self, x: float) -> float:
        """Distance normalized by radius (0 at center, ~1 at boundary)."""
        r = self.radius
        if r <= 0:
            return 0.0
        return self.distance(x) / r

    def contains(self, x: float) -> bool:
        """True if x lies within the basin boundaries (inclusive)."""
        return self.boundary_lower <= float(x) <= self.boundary_upper

    def boundary_margin(self, x: float) -> float:
        """Distance from x to the nearest boundary (0 if inside)."""
        if self.contains(x):
            lower_gap = float(x) - self.boundary_lower
            upper_gap = self.boundary_upper - float(x)
            return min(lower_gap, upper_gap)
        return min(abs(float(x) - self.boundary_lower), abs(float(x) - self.boundary_upper))

    def stability_score(self) -> float:
        """
        Heuristic stability metric in [0, 1].

        Combines curvature (stiffness), tightness (narrow radius), and sample support.
        """
        curvature_component = max(min(self.curvature, 1.0), 0.0)
        radius_component = 1.0 / (1.0 + max(self.radius, 0.0))
        support_component = min(self.sample_count / 20.0, 1.0)
        score = 0.4 * curvature_component + 0.4 * radius_component + 0.2 * support_component
        return max(min(score, 1.0), 0.0)

    def to_dict(self) -> Dict:
        """Legacy dictionary shape used by downstream serialization/DB writes."""
        base = {
            "center": float(self.center),
            "boundary_upper": float(self.boundary_upper),
            "boundary_lower": float(self.boundary_lower),
            "width": float(self.width),
            "sample_count": int(self.sample_count),
            "curvature": float(self.curvature),
        }
        # Preserve legacy shape while allowing 2D metadata when present.
        if self.domain_dim > 1:
            base["center_vector"] = list(self.center_vector) if self.center_vector else None
            base["width_vector"] = list(self.width_vector) if self.width_vector else None
            base["covariance"] = self.covariance
            base["domain_dim"] = int(self.domain_dim)
        return base

    @classmethod
    def from_geometry(
        cls,
        *,
        center: float,
        std: float,
        boundary_sigma: float,
        sample_count: int,
        curvature: float,
        domain_dim: int = 1,
        center_vector: Optional[List[float]] = None,
        width_vector: Optional[List[float]] = None,
        covariance: Optional[List[List[float]]] = None,
    ) -> "ResidualBasin":
        boundary_upper = center + (boundary_sigma * std)
        boundary_lower = center - (boundary_sigma * std)
        width = boundary_upper - boundary_lower
        variance = std * std
        return cls(
            center=float(center),
            width=float(width),
            boundary_upper=float(boundary_upper),
            boundary_lower=float(boundary_lower),
            curvature=float(curvature),
            sample_count=int(sample_count),
            variance=float(variance),
            domain_dim=int(domain_dim),
            center_vector=center_vector,
            width_vector=width_vector,
            covariance=covariance,
        )


@dataclass(frozen=True)
class BasinEvolution:
    center_delta: float
    radius_delta: float
    width_ratio: float
    compression: float
    expansion: float
    stability_delta: float


def compare_basins(before: ResidualBasin, after: ResidualBasin) -> BasinEvolution:
    """
    Compute basin evolution diagnostics between two immutable basins.

    Returns deltas and compression/expansion magnitudes without mutating inputs.
    """
    center_delta = float(after.center - before.center)
    radius_delta = float(after.radius - before.radius)
    width_ratio = float(after.width / before.width) if before.width != 0 else float("inf")

    compression = float(max(0.0, 1.0 - width_ratio)) if width_ratio != float("inf") else 0.0
    expansion = float(max(0.0, width_ratio - 1.0)) if width_ratio != float("inf") else 0.0

    stability_delta = float(after.stability_score() - before.stability_score())

    return BasinEvolution(
        center_delta=center_delta,
        radius_delta=radius_delta,
        width_ratio=width_ratio,
        compression=compression,
        expansion=expansion,
        stability_delta=stability_delta,
    )
