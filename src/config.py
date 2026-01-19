from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    # Units: micrometers (um)
    container_diameter_um: float = 1000.0
    container_radius_um: float = 500.0

    # Particle equivalent diameter distribution (um)
    d_min_um: float = 30.0
    d_max_um: float = 90.0

    # How many particles to attempt
    n_particles_target: int = 400

    # Gravity/sedimentation heuristic parameters (dimensionless)
    max_drop_steps: int = 600
    lateral_step_um: float = 3.0
    vertical_step_um: float = 2.5
    settle_eps_um: float = 0.8

    # Irregularity controls for single-particle geometry
    # Create particles as radial function on sphere: r(θ,φ)=r0*(1+Σ a_k cos(...))
    n_bumps: int = 6
    bump_amp_range: tuple[float, float] = (0.04, 0.14)

    # Mesh resolution for particles
    sphere_subdivisions: int = 3

    # Contact/overlap tolerance
    overlap_tol_um: float = 0.2

    # Output
    out_dir: str = "outputs"
