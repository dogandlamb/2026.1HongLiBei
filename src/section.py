from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely.geometry as geom

from .packing import PackResult


@dataclass
class SectionData:
    z_um: float
    container_radius_um: float
    polygons: list[geom.Polygon]

    @property
    def centroids_xy(self) -> np.ndarray:
        if not self.polygons:
            return np.zeros((0, 2), dtype=float)
        pts = np.array([[p.centroid.x, p.centroid.y] for p in self.polygons], dtype=float)
        return pts


def _circle_polygon(radius: float, n: int = 256) -> geom.Polygon:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.c_[radius * np.cos(th), radius * np.sin(th)]
    return geom.Polygon(pts)


def extract_horizontal_section(pack: PackResult, z_um: float) -> SectionData:
    """Intersect each particle mesh with the plane z=z_um.

    Returns 2D polygons in the XY plane clipped to the container circle.
    """

    circle = _circle_polygon(pack.container_radius_um)

    polys: list[geom.Polygon] = []

    for p in pack.particles:
        # Quick reject by bounding sphere
        r = p.bounding_radius_um
        if abs(float(p.center_um[2] - z_um)) > r:
            continue

        # trimesh section gives a Path3D curve(s)
        sec = p.mesh.section(plane_origin=[0, 0, z_um], plane_normal=[0, 0, 1])
        if sec is None:
            continue

        # Use 3D discrete polylines and project to world XY.
        # This avoids local 2D coordinate frames returned by to_2D/to_planar.
        try:
            disc3 = getattr(sec, "discrete", None)
            if disc3 is None:
                polylines3 = []
            elif callable(disc3):
                polylines3 = list(disc3())
            else:
                polylines3 = list(disc3)
        except Exception:
            polylines3 = []

        for pts3 in polylines3:
            try:
                pts3 = np.asarray(pts3, dtype=float)
                if pts3.ndim != 2 or pts3.shape[0] < 3:
                    continue
                pts = pts3[:, :2]
                if not np.allclose(pts[0], pts[-1]):
                    pts = np.vstack([pts, pts[0]])

                poly = geom.Polygon(pts)
                if (not poly.is_valid) or poly.area <= 0:
                    continue

                poly = poly.intersection(circle)
                if poly.is_empty:
                    continue

                if isinstance(poly, geom.Polygon):
                    if poly.area > 1e-6:
                        polys.append(poly)
                else:
                    for g in getattr(poly, "geoms", []):
                        if isinstance(g, geom.Polygon) and g.area > 1e-6:
                            polys.append(g)
            except Exception:
                continue

    return SectionData(z_um=float(z_um), container_radius_um=pack.container_radius_um, polygons=polys)
