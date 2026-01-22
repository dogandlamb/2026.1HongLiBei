import math
import os
import random
import sys
import importlib.util
from dataclasses import dataclass

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import trimesh

# Load local src/gjk.py explicitly to avoid name collisions with any installed 'gjk' package.
_SRC_DIR = os.path.dirname(__file__)
_GJK_PATH = os.path.join(_SRC_DIR, 'gjk.py')
_spec = importlib.util.spec_from_file_location('local_gjk', _GJK_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f'Failed to load local GJK module at: {_GJK_PATH}')
gjk_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gjk_module)
import particle_generator
import size_distribution
import voronoi_generator


@dataclass
class PackingParams:
    # Container (paper): Dc=1.0mm, Hc=20.0mm
    container_radius_um: float = 500.0
    container_height_um: float = 20000.0

    # Particle generation (paper)
    bounds: tuple = ((0.0, 200.0), (0.0, 200.0), (0.0, 200.0))
    expected_seed_count: int = 50
    diameter_mu_ln: float = 4.0
    diameter_sigma_ln: float = 0.3
    diameter_range_um: tuple = (30.0, 90.0)
    perlin_amplitude_um: float = 2.5

    # Simulation
    dt_s: float = 1e-7  # 0.1 μs
    max_steps: int = 5000
    cell_size_um: float = 120.0


class Particle:
    def __init__(self, particle_id: int, vertices_um: np.ndarray):
        self.id = particle_id

        self.local_vertices = np.asarray(vertices_um, dtype=float)
        self.radius_approx = float(np.max(np.linalg.norm(self.local_vertices, axis=1)))

        # For demo stability we keep a virtual mass.
        self.mass = 1.0
        self.inertia = np.eye(3) * 0.4 * self.mass * (self.radius_approx**2)
        self.inv_inertia = np.linalg.inv(self.inertia)
        self.inv_mass = 1.0 / self.mass

        self.position = np.zeros(3)
        self.rotation = np.eye(3)
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.force = np.zeros(3)
        self.torque = np.zeros(3)

        self.aabb_min = np.zeros(3)
        self.aabb_max = np.zeros(3)
        self.update_aabb()

    def get_world_vertices(self) -> np.ndarray:
        return np.dot(self.local_vertices, self.rotation.T) + self.position

    def update_aabb(self) -> None:
        pts = self.get_world_vertices()
        self.aabb_min = np.min(pts, axis=0)
        self.aabb_max = np.max(pts, axis=0)

    def integrate(self, dt: float) -> None:
        acc = self.force * self.inv_mass
        self.velocity += acc * dt
        self.position += self.velocity * dt

        ang_acc = np.dot(self.inv_inertia, self.torque)
        self.angular_velocity += ang_acc * dt

        # Mild damping for numerical stability
        self.velocity *= 0.999
        self.angular_velocity *= 0.999

        theta = np.linalg.norm(self.angular_velocity) * dt
        if theta > 1e-8:
            axis = self.angular_velocity / (theta / dt)
            K = np.array(
                [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
            )
            R_inc = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
            self.rotation = np.dot(R_inc, self.rotation)
            try:
                u, _, vt = np.linalg.svd(self.rotation)
                self.rotation = np.dot(u, vt)
            except np.linalg.LinAlgError:
                self.rotation = np.eye(3)
                self.angular_velocity[:] = 0

        self.update_aabb()
        self.force[:] = 0
        self.torque[:] = 0


class SpatialHash:
    def __init__(self, cell_size: float):
        self.cell_size = float(cell_size)
        self.grid: dict[tuple[int, int, int], list[Particle]] = {}

    def clear(self) -> None:
        self.grid = {}

    def insert(self, particle: Particle) -> None:
        min_idx = np.floor(particle.aabb_min / self.cell_size).astype(int)
        max_idx = np.floor(particle.aabb_max / self.cell_size).astype(int)

        for x in range(min_idx[0], max_idx[0] + 1):
            for y in range(min_idx[1], max_idx[1] + 1):
                for z in range(min_idx[2], max_idx[2] + 1):
                    key = (x, y, z)
                    self.grid.setdefault(key, []).append(particle)

    def get_candidates(self) -> list[tuple[Particle, Particle]]:
        candidates: set[tuple[int, int]] = set()
        pairs: list[tuple[Particle, Particle]] = []

        for bucket in self.grid.values():
            n = len(bucket)
            for i in range(n):
                for j in range(i + 1, n):
                    p1, p2 = bucket[i], bucket[j]
                    a, b = (p1.id, p2.id) if p1.id < p2.id else (p2.id, p1.id)
                    if (a, b) in candidates:
                        continue
                    candidates.add((a, b))
                    pairs.append((p1, p2) if p1.id < p2.id else (p2, p1))

        return pairs


class Container:
    def __init__(self, radius_um: float, height_um: float):
        self.radius = float(radius_um)
        self.height = float(height_um)

    def check_collision_hertz(self, particle: Particle) -> None:
        pts = particle.get_world_vertices()

        # Floor z=0
        min_z = float(np.min(pts[:, 2]))
        if min_z < 0:
            depth = -min_z
            normal = np.array([0.0, 0.0, 1.0])
            fn, ft = calculate_hertz_mindlin_force(None, particle, depth, normal, is_wall=True)
            particle.force += fn + ft

        # Cylinder wall
        radii = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        max_r_idx = int(np.argmax(radii))
        max_r = float(radii[max_r_idx])
        if max_r > self.radius:
            depth = max_r - self.radius
            normal_vec = -np.array([pts[max_r_idx, 0], pts[max_r_idx, 1], 0.0])
            nlen = np.linalg.norm(normal_vec)
            if nlen > 1e-12:
                normal_vec /= nlen
            else:
                normal_vec = np.array([0.0, 0.0, 0.0])

            fn, ft = calculate_hertz_mindlin_force(None, particle, depth, normal_vec, is_wall=True)
            r_vec = pts[max_r_idx] - particle.position
            particle.force += fn + ft
            particle.torque += np.cross(r_vec, fn + ft)


def calculate_hertz_mindlin_force(p_other, p_current, depth, normal, is_wall: bool = False):
    # Simplified Hertz-Mindlin (structure matches paper; constants tuned for this demo scale)
    kn = 2e5
    gn = 50
    mu = 0.5

    v1 = p_current.velocity
    v2 = p_other.velocity if (p_other is not None and not is_wall) else np.zeros(3)
    v_rel = v1 - v2
    vn = float(np.dot(v_rel, normal))

    if depth < 0:
        depth = 0
    fn_mag = kn * (depth ** 1.5) - gn * math.sqrt(depth) * vn
    if fn_mag < 0:
        fn_mag = 0
    fn_vec = fn_mag * normal

    vt = v_rel - vn * normal
    ft_mag = 0.5 * np.linalg.norm(vt) * fn_mag
    if ft_mag > mu * fn_mag:
        ft_mag = mu * fn_mag

    vt_norm = np.linalg.norm(vt)
    if vt_norm > 1e-9:
        ft_vec = -(vt / vt_norm) * ft_mag
    else:
        ft_vec = np.zeros(3)

    return fn_vec, ft_vec


def _random_point_in_circle(r_max: float) -> tuple[float, float]:
    u = random.random()
    r = r_max * math.sqrt(u)
    theta = random.random() * 2.0 * math.pi
    return r * math.cos(theta), r * math.sin(theta)


def run_simulation(params: PackingParams | None = None):
    if params is None:
        params = PackingParams()

    # Allow overrides via env vars
    num_particles = int(os.environ.get('PACKING_N', '500'))
    max_steps = int(os.environ.get('PACKING_MAX_STEPS', str(params.max_steps)))
    dt = float(os.environ.get('PACKING_DT', str(params.dt_s)))

    container = Container(params.container_radius_um, params.container_height_um)

    intensity = float(params.expected_seed_count) / float(200.0 ** 3)
    seeds = voronoi_generator.generate_poisson_seeds(params.bounds, intensity, random_seed=42)
    polyhedrons = voronoi_generator.generate_voronoi_polyhedrons(seeds)
    keys = list(polyhedrons.keys())
    if not keys:
        raise RuntimeError('No bounded Voronoi cells generated')

    particles: list[Particle] = []

    for i in range(num_particles):
        k = keys[i % len(keys)]
        verts0 = polyhedrons[k]

        target_d = size_distribution.sample_truncated_lognormal(
            mu_ln=params.diameter_mu_ln,
            sigma_ln=params.diameter_sigma_ln,
            size_range=params.diameter_range_um,
        )

        try:
            verts = voronoi_generator.scale_polyhedron_to_eq_diameter(verts0, target_d)
            verts = np.asarray(verts, dtype=float)
            verts = particle_generator.perlin_noise_modification(verts.tolist(), params.perlin_amplitude_um)
            verts = voronoi_generator.scale_polyhedron_to_eq_diameter(verts, target_d)
            verts = np.asarray(verts, dtype=float)

            p = Particle(i, verts)
            x, y = _random_point_in_circle(container.radius - target_d / 2.0)
            # Optimize start height: Start closer to bottom (100um) instead of container.height (20000um)
            # Scale spacing by 1.2 * diameter to avoid initial overlaps
            z = 100.0 + i * (1.2 * target_d)
            p.position = np.array([x, y, z], dtype=float)
            p.update_aabb()
            particles.append(p)
        except Exception:
            continue

    spatial_hash = SpatialHash(cell_size=params.cell_size_um)

    gravity = np.array([0.0, 0.0, -9.8e6])

    for step in range(1, max_steps + 1):
        spatial_hash.clear()
        for p in particles:
            spatial_hash.insert(p)
        candidates = spatial_hash.get_candidates()

        total_energy = 0.0

        for p in particles:
            p.force += gravity * p.mass
            container.check_collision_hertz(p)
            total_energy += 0.5 * p.mass * float(np.dot(p.velocity, p.velocity))

        for p1, p2 in candidates:
            # AABB prune
            if (
                p1.aabb_max[0] < p2.aabb_min[0]
                or p1.aabb_min[0] > p2.aabb_max[0]
                or p1.aabb_max[1] < p2.aabb_min[1]
                or p1.aabb_min[1] > p2.aabb_max[1]
                or p1.aabb_max[2] < p2.aabb_min[2]
                or p1.aabb_min[2] > p2.aabb_max[2]
            ):
                continue

            pts1 = p1.get_world_vertices()
            pts2 = p2.get_world_vertices()
            is_col, simplex = gjk_module.gjk(pts1, pts2)
            if not is_col:
                continue

            try:
                depth, normal = gjk_module.epa(pts1, pts2, simplex)
            except Exception:
                continue

            fn, ft = calculate_hertz_mindlin_force(p2, p1, depth, normal)
            p1.force += fn + ft
            p2.force -= (fn + ft)

        for p in particles:
            p.integrate(dt)

        if step % 100 == 0:
            print(f'Step {step}: Total KE = {total_energy:.4e}')
            if total_energy < 1e-5 and step > 200:
                print('Equilibrium reached.')
                break

    return particles, container


def visualize_packing(particles, container):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    z = np.linspace(0, min(container.height, 5000.0), 2)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = container.radius * np.cos(theta_grid)
    y_grid = container.radius * np.sin(theta_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.1, color='gray')
    ax.plot(x_grid[0], y_grid[0], z_grid[0], color='k')

    colors = plt.cm.jet(np.linspace(0, 1, len(particles)))

    for i, p in enumerate(particles):
        verts = p.get_world_vertices()
        try:
            hull = ConvexHull(verts)
            faces = [verts[s] for s in hull.simplices]
            poly = Poly3DCollection(faces, alpha=0.8, edgecolor='k', facecolor=colors[i])
            ax.add_collection3d(poly)
        except Exception:
            continue

    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_zlabel('Z (um)')
    ax.set_xlim(-container.radius - 100, container.radius + 100)
    ax.set_ylim(-container.radius - 100, container.radius + 100)
    ax.set_zlim(0, min(container.height, 5000.0))
    ax.set_box_aspect([1, 1, 1])
    ax.set_title('Random Packing in Cylindrical Container (DEM+GJK)')

    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/packing_simulation_plot.png')
    if os.environ.get('PACKING_SHOW', '0') == '1':
        plt.show()
    else:
        plt.close(fig)


def save_to_glb(particles, container, filename='outputs/output.glb'):
    print('Generating GLB file...')
    scene = trimesh.Scene()

    cyl = trimesh.creation.cylinder(radius=container.radius, height=container.height, sections=64)
    cyl.apply_translation([0, 0, container.height / 2])
    cyl.visual.face_colors = [200, 200, 200, 50]
    scene.add_geometry(cyl)

    for p in particles:
        verts = p.get_world_vertices()
        try:
            hull = ConvexHull(verts)
            mesh = trimesh.Trimesh(vertices=verts, faces=hull.simplices)
            mesh.visual.face_colors = trimesh.visual.random_color()
            scene.add_geometry(mesh)
        except Exception:
            continue

    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    scene.export(filename)
    print(f'GLB file saved to: {filename}')
