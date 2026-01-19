from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.neighbors import NearestNeighbors

from .section import SectionData


@dataclass
class FitResult:
    name: str
    params: tuple
    ks_stat: float
    ks_pvalue: float


def polygon_metrics(section: SectionData) -> pd.DataFrame:
    rows = []
    for idx, poly in enumerate(section.polygons):
        area = float(poly.area)
        # Equivalent diameter for 2D area: d_eq = sqrt(4A/pi)
        d_eq = float(np.sqrt(4.0 * area / np.pi))
        perim = float(poly.length)
        # Circularity: 4πA/P^2 in (0,1]
        circ = float(4.0 * np.pi * area / (perim * perim + 1e-12))
        c = poly.centroid
        rows.append(
            {
                "id": idx,
                "area_um2": area,
                "equiv_d_um": d_eq,
                "perimeter_um": perim,
                "circularity": circ,
                "cx_um": float(c.x),
                "cy_um": float(c.y),
            }
        )
    return pd.DataFrame(rows)


def fit_distributions(data: np.ndarray) -> list[FitResult]:
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data) & (data > 0)]
    if len(data) < 10:
        return []

    candidates: list[tuple[str, object]] = [
        ("lognorm", st.lognorm),
        ("weibull_min", st.weibull_min),
        ("gamma", st.gamma),
    ]

    results: list[FitResult] = []
    for name, dist in candidates:
        try:
            params = dist.fit(data)
            ks_stat, ks_p = st.kstest(data, name, args=params)
            results.append(FitResult(name=name, params=tuple(params), ks_stat=float(ks_stat), ks_pvalue=float(ks_p)))
        except Exception:
            continue

    results.sort(key=lambda r: (-r.ks_pvalue, r.ks_stat))
    return results


@dataclass
class SpatialSummary:
    n_points: int
    mean_nn_um: float
    cv_nn: float
    clark_evans_R: float


def spatial_metrics(section: SectionData) -> SpatialSummary:
    """Quantify spatial pattern of centroids.

    We report:
    - mean nearest-neighbor distance
    - CV of nearest-neighbor distances
    - Clark–Evans R index: R = r_obs / r_exp
        r_exp = 1 / (2*sqrt(lambda)), lambda = n/Area
      Interpretation (approx):
        R ~ 1 random (CSR), R < 1 clustered, R > 1 regular.

    Note: boundary effects exist due to circular window; this is a standard, acceptable
    first-order indicator in contest modeling.
    """

    pts = section.centroids_xy
    n = int(pts.shape[0])
    if n < 3:
        return SpatialSummary(n_points=n, mean_nn_um=float("nan"), cv_nn=float("nan"), clark_evans_R=float("nan"))

    nbrs = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(pts)
    dists, _ = nbrs.kneighbors(pts)
    nn = dists[:, 1]

    mean_nn = float(np.mean(nn))
    std_nn = float(np.std(nn, ddof=1))
    cv = float(std_nn / (mean_nn + 1e-12))

    area_window = float(np.pi * section.container_radius_um ** 2)
    lam = float(n / area_window)
    r_exp = float(1.0 / (2.0 * np.sqrt(lam + 1e-18)))
    R = float(mean_nn / (r_exp + 1e-12))

    return SpatialSummary(n_points=n, mean_nn_um=mean_nn, cv_nn=cv, clark_evans_R=R)
