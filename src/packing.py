from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from .config import SimConfig
from .particle import Particle, make_irregular_particle_mesh, rotate_mesh, transform_mesh
from .random_utils import random_unit_quaternion, sample_equivalent_diameter_um


@dataclass
class PackResult:
    particles: list[Particle]
    container_radius_um: float
    height_um: float


class _SpatialHash3D:
    def __init__(self, cell_size_um: float):
        self.cell = float(cell_size_um)
        self.map: dict[tuple[int, int, int], list[int]] = {}

    def _key(self, p: np.ndarray) -> tuple[int, int, int]:
        c = self.cell
        return (int(np.floor(p[0] / c)), int(np.floor(p[1] / c)), int(np.floor(p[2] / c)))

    def insert(self, center: np.ndarray, index: int) -> None:
        k = self._key(center)
        self.map.setdefault(k, []).append(index)

    def query_neighbor_indices(self, center: np.ndarray) -> list[int]:
        ix, iy, iz = self._key(center)
        out: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    out.extend(self.map.get((ix + dx, iy + dy, iz + dz), []))
        return out


def _inside_cylinder_xy(xy: np.ndarray, R: float, margin: float = 0.0) -> bool:
    return float(xy[0] ** 2 + xy[1] ** 2) <= float((R - margin) ** 2)


def _fast_sphere_overlap(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float, tol: float) -> bool:
    d2 = float(np.sum((c1 - c2) ** 2))
    return d2 < float((r1 + r2 - tol) ** 2)


def simulate_random_packing(cfg: SimConfig, seed: int | None = 0) -> PackResult:
    """Heuristic gravity deposition packing in a cylinder.

    This is a modeling-oriented simulation (not DEM): sequentially drop particles from above,
    apply random lateral drift while moving downward, and accept the first non-overlapping
    settled position. It captures boundary effect + random microstructure, sufficient for
    statistical section analysis.
    """

    rng = np.random.default_rng(seed)

    particles: list[Particle] = []
    R = cfg.container_radius_um

    # Use a conservative cell size so that potential overlaps are always in nearby cells.
    grid = _SpatialHash3D(cell_size_um=max(cfg.d_max_um, 1.0))

    # Estimate height to aim for: roughly achieve moderate fill. Heuristic.
    # Increase automatically as particles accumulate.
    current_height = 0.0

    for i in range(cfg.n_particles_target):
        if (i + 1) % 50 == 0:
            print(f"  placing particle {i+1}/{cfg.n_particles_target} (accepted={len(particles)})")
        equiv_d = sample_equivalent_diameter_um(rng, cfg.d_min_um, cfg.d_max_um, mode="uniform")
        base_mesh = make_irregular_particle_mesh(
            rng,
            equiv_d_um=equiv_d,
            n_bumps=cfg.n_bumps,
            bump_amp_range=cfg.bump_amp_range,
            sphere_subdivisions=cfg.sphere_subdivisions,
        )
        q = random_unit_quaternion(rng)
        rot_mesh = rotate_mesh(base_mesh, q)
        bs_r = float(rot_mesh.bounding_sphere.primitive.radius)

        # spawn above the current top
        z0 = current_height + 4.0 * equiv_d
        # choose a start xy inside cylinder
        for _ in range(60):
            xy = rng.uniform(low=-R, high=R, size=(2,))
            if _inside_cylinder_xy(xy, R, margin=bs_r):
                break
        else:
            continue

        pos = np.array([xy[0], xy[1], z0], dtype=float)

        accepted = False
        # perform downward settling with small random drift
        for step in range(cfg.max_drop_steps):
            # propose move
            drift = rng.normal(scale=1.0, size=(2,))
            drift_norm = np.linalg.norm(drift) + 1e-12
            drift = drift / drift_norm
            dxy = drift * cfg.lateral_step_um
            dz = -cfg.vertical_step_um

            cand = pos.copy()
            cand[:2] += dxy
            raw_z = float(cand[2] + dz)
            cand[2] = raw_z

            # ground plane at z=0
            if cand[2] - bs_r < 0:
                cand[2] = bs_r

            # keep inside cylinder
            if not _inside_cylinder_xy(cand[:2], R, margin=bs_r):
                # reflect drift: pull back toward center
                cand[:2] = pos[:2] * 0.85

            # collision check with existing (local neighbors only)
            collided = False
            neighbor_ids = grid.query_neighbor_indices(cand)
            for j in neighbor_ids:
                p = particles[j]
                if _fast_sphere_overlap(cand, bs_r, p.center_um, p.bounding_radius_um, cfg.overlap_tol_um):
                    collided = True
                    break

            if not collided:
                pos = cand

                # If we attempted to go below the floor, we have contacted the bottom.
                if raw_z - bs_r <= 0:
                    accepted = True
                    break
            else:
                # try to settle: if last move caused collision, accept current pos as settled
                # if it's still colliding at current pos, reject entirely
                bad = False
                neighbor_ids2 = grid.query_neighbor_indices(pos)
                for j in neighbor_ids2:
                    p = particles[j]
                    if _fast_sphere_overlap(pos, bs_r, p.center_um, p.bounding_radius_um, cfg.overlap_tol_um):
                        bad = True
                        break
                if not bad:
                    accepted = True
                break

            # early stop if tiny motion
            if abs(dz) < cfg.settle_eps_um:
                accepted = True
                break

        if accepted:
            final_mesh = transform_mesh(rot_mesh, np.eye(3), pos)
            particles.append(Particle(mesh=final_mesh, equiv_d_um=equiv_d, center_um=pos, bounding_radius_um=bs_r))
            grid.insert(pos, len(particles) - 1)
            current_height = max(current_height, float(pos[2] + bs_r))

    return PackResult(particles=particles, container_radius_um=R, height_um=float(current_height))
