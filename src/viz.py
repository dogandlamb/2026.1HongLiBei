from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh

from .packing import PackResult


def ensure_out_dir(out_dir: str) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def export_packing_mesh(pack: PackResult, out_dir: str, filename: str = "packing.glb") -> Path:
    out = ensure_out_dir(out_dir)

    scene = trimesh.Scene()
    for idx, p in enumerate(pack.particles):
        mesh = p.mesh.copy()
        color = trimesh.visual.random_color()
        mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
        scene.add_geometry(mesh, node_name=f"p{idx}")

    # add container wireframe
    # cylinder axis along z
    cyl = trimesh.creation.cylinder(
        radius=pack.container_radius_um,
        height=max(pack.height_um, 1.0),
        sections=64,
    )
    cyl.apply_translation([0, 0, max(pack.height_um, 1.0) / 2.0])
    cyl.visual.face_colors = [0, 0, 0, 15]
    scene.add_geometry(cyl, node_name="container")

    out_path = out / filename
    scene.export(out_path)
    return out_path


def quick_topdown_plot(pack: PackResult, out_dir: str, filename: str = "topdown.png") -> Path:
    out = ensure_out_dir(out_dir)

    xy = np.array([p.center_um[:2] for p in pack.particles], dtype=float)
    s = np.array([p.equiv_d_um for p in pack.particles], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.scatter(xy[:, 0], xy[:, 1], s=(s ** 2) * 0.2, alpha=0.7, edgecolor="none")

    th = np.linspace(0, 2 * np.pi, 400)
    R = pack.container_radius_um
    ax.plot(R * np.cos(th), R * np.sin(th), color="black", linewidth=1.0)

    ax.set_aspect("equal")
    ax.set_title("Particle centers (top-down)")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.grid(True, alpha=0.2)

    out_path = out / filename
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
