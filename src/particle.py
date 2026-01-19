from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class Particle:
    mesh: trimesh.Trimesh
    equiv_d_um: float
    center_um: np.ndarray  # (3,)
    bounding_radius_um: float


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    # q=(w,x,y,z)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def make_irregular_particle_mesh(
    rng: np.random.Generator,
    equiv_d_um: float,
    n_bumps: int = 6,
    bump_amp_range: tuple[float, float] = (0.04, 0.14),
    sphere_subdivisions: int = 3,
) -> trimesh.Trimesh:
    """Create a non-spherical particle as a radial deformation of an icosphere.

    Geometry model:
        Start with a sphere of radius r0=equiv_d/2.
        For each vertex direction u on sphere, set radius:
            r(u)=r0*(1 + Σ a_k * cos( k·angle(u, u_k) + phi_k ))
        Then vertex position = r(u)*u.

    This yields an irregular, angular-ish surface while keeping a single closed mesh.
    """

    r0 = float(equiv_d_um) / 2.0
    sphere = trimesh.creation.icosphere(subdivisions=int(sphere_subdivisions), radius=1.0)

    # random bump directions on sphere
    dirs = rng.normal(size=(n_bumps, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12

    amps = rng.uniform(bump_amp_range[0], bump_amp_range[1], size=(n_bumps,))
    phases = rng.uniform(0.0, 2 * np.pi, size=(n_bumps,))
    freqs = rng.integers(low=1, high=4, size=(n_bumps,))

    V = sphere.vertices.copy()
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    U = V / (norms + 1e-12)  # unit directions

    # compute angular similarity with each bump direction
    # cos(angle)=u·d
    cosang = U @ dirs.T  # (nV, n_bumps)
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.arccos(cosang)

    # radial factor
    bump = np.zeros((U.shape[0],), dtype=float)
    for k in range(n_bumps):
        bump += amps[k] * np.cos(freqs[k] * ang[:, k] + phases[k])

    r = r0 * (1.0 + bump)
    r = np.maximum(r, 0.35 * r0)

    V_new = U * r.reshape(-1, 1)
    mesh = trimesh.Trimesh(vertices=V_new, faces=sphere.faces, process=True)

    # Cleanup: trimesh APIs vary across versions; keep this conservative.
    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    try:
        mesh.merge_vertices()
    except Exception:
        pass
    try:
        mesh.process(validate=True)
    except Exception:
        pass

    # Important: center mesh at origin so rotations/translations behave as expected.
    try:
        c = np.array(mesh.bounding_sphere.center, dtype=float)
        mesh.apply_translation(-c)
    except Exception:
        pass

    return mesh


def transform_mesh(mesh: trimesh.Trimesh, R: np.ndarray, t: np.ndarray) -> trimesh.Trimesh:
    m = mesh.copy()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    m.apply_transform(T)
    return m


def rotate_mesh(mesh: trimesh.Trimesh, q: np.ndarray) -> trimesh.Trimesh:
    R = _quaternion_to_matrix(q)
    return transform_mesh(mesh, R, np.zeros(3))
