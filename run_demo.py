from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import SimConfig
from src.packing import simulate_random_packing
from src.section import extract_horizontal_section
from src.analysis import polygon_metrics, fit_distributions, spatial_metrics
from src.viz import export_packing_mesh, quick_topdown_plot
from src.plots import plot_section_polygons, plot_hist_with_fit, plot_centroids_scatter


def main() -> None:
    cfg = SimConfig(
        n_particles_target=180,
        sphere_subdivisions=3,
        n_bumps=7,
        max_drop_steps=400,
        vertical_step_um=4.0,
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Simulating 3D packing...")
    pack = simulate_random_packing(cfg, seed=42)
    print(f"Packed particles: {len(pack.particles)}")
    print(f"Estimated height: {pack.height_um:.1f} um")

    glb = export_packing_mesh(pack, cfg.out_dir, filename="packing.glb")
    top = quick_topdown_plot(pack, cfg.out_dir, filename="topdown.png")
    print(f"3D export: {glb}")
    print(f"Top-down plot: {top}")

    print("Extracting horizontal section...")
    # Choose a section height that yields many intersections
    if pack.height_um > 1:
        candidate_z = np.linspace(0.15 * pack.height_um, 0.85 * pack.height_um, 9)
    else:
        candidate_z = np.array([0.0])

    best_sec = None
    best_n = -1
    for z in candidate_z:
        s = extract_horizontal_section(pack, z_um=float(z))
        if len(s.polygons) > best_n:
            best_n = len(s.polygons)
            best_sec = s

    sec = best_sec
    assert sec is not None
    print(f"Section polygons: {len(sec.polygons)} at z={sec.z_um:.1f} um")

    sec_plot = plot_section_polygons(sec, cfg.out_dir, filename="section.png")
    print(f"Section plot: {sec_plot}")

    df = polygon_metrics(sec)
    df_path = out_dir / "section_metrics.csv"
    df.to_csv(df_path, index=False)
    print(f"Metrics CSV: {df_path}")

    if len(df) >= 10:
        fits_area = fit_distributions(df["area_um2"].to_numpy())
        fits_d = fit_distributions(df["equiv_d_um"].to_numpy())
        best_area = fits_area[0] if fits_area else None
        best_d = fits_d[0] if fits_d else None

        plot_hist_with_fit(
            df["area_um2"].to_numpy(),
            best_area,
            cfg.out_dir,
            filename="area_fit.png",
            title="Section area distribution (fit)",
            xlabel="area (um^2)",
        )
        plot_hist_with_fit(
            df["equiv_d_um"].to_numpy(),
            best_d,
            cfg.out_dir,
            filename="deq_fit.png",
            title="Equivalent diameter distribution (fit)",
            xlabel="d_eq (um)",
        )

        plot_centroids_scatter(df, sec.container_radius_um, cfg.out_dir, filename="centroids.png")

        spatial = spatial_metrics(sec)
        summary = {
            "packed_particles": len(pack.particles),
            "height_um": pack.height_um,
            "section_z_um": sec.z_um,
            "n_sections": len(sec.polygons),
            "best_fit_area": None if best_area is None else best_area.__dict__,
            "best_fit_deq": None if best_d is None else best_d.__dict__,
            "spatial": spatial.__dict__,
        }

        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary JSON: {out_dir / 'summary.json'}")
    else:
        print("Not enough section polygons for distribution fitting (need >=10).")


if __name__ == "__main__":
    main()
