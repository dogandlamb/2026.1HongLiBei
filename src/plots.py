from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

from .analysis import FitResult
from .section import SectionData


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_section_polygons(section: SectionData, out_dir: str, filename: str = "section.png") -> Path:
    out = ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)

    th = np.linspace(0, 2 * np.pi, 400)
    R = section.container_radius_um
    ax.plot(R * np.cos(th), R * np.sin(th), color="black", linewidth=1.0)

    for poly in section.polygons:
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.7, linewidth=0.5)

    ax.set_aspect("equal")
    ax.set_title(f"Horizontal section at z={section.z_um:.1f} um")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.grid(True, alpha=0.2)

    out_path = out / filename
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_hist_with_fit(
    data: np.ndarray,
    fit: FitResult | None,
    out_dir: str,
    filename: str,
    title: str,
    xlabel: str,
) -> Path:
    out = ensure_dir(out_dir)
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=180)
    ax.hist(data, bins=30, density=True, alpha=0.7, color="#4C78A8")

    if fit is not None:
        x = np.linspace(np.min(data), np.max(data), 300)
        dist = getattr(st, fit.name)
        y = dist.pdf(x, *fit.params)
        ax.plot(x, y, color="#F58518", linewidth=2.0, label=f"{fit.name}, KS p={fit.ks_pvalue:.3g}")
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.2)

    out_path = out / filename
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_centroids_scatter(df: pd.DataFrame, container_radius_um: float, out_dir: str, filename: str = "centroids.png") -> Path:
    out = ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    ax.scatter(df["cx_um"], df["cy_um"], s=12, alpha=0.8)

    th = np.linspace(0, 2 * np.pi, 400)
    R = container_radius_um
    ax.plot(R * np.cos(th), R * np.sin(th), color="black", linewidth=1.0)

    ax.set_aspect("equal")
    ax.set_title("Section centroids")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.grid(True, alpha=0.2)

    out_path = out / filename
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
