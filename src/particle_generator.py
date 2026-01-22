import math
import sys
import numpy as np
from scipy.spatial import ConvexHull

import size_distribution
import voronoi_generator


def laplacian_smoothing(vertices, iterations=2, alpha=0.5):
    """
    Apply Laplacian smoothing to the mesh vertices.
    """
    verts = np.array(vertices)
    if len(verts) < 4: return vertices
    
    for _ in range(iterations):
        try:
            hull = ConvexHull(verts)
            # Build adjacency
            adj = {i: set() for i in range(len(verts))}
            for simplex in hull.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        u, v = simplex[i], simplex[j]
                        adj[u].add(v)
                        adj[v].add(u)
            
            new_verts = verts.copy()
            for i in range(len(verts)):
                neighbors = list(adj[i])
                if not neighbors: continue
                neighbor_sum = np.sum(verts[neighbors], axis=0)
                target = neighbor_sum / len(neighbors)
                # Formula: v_new = v + alpha * (avg_neighbor - v)
                new_verts[i] = verts[i] + alpha * (target - verts[i])
            verts = new_verts
        except Exception as e:
            print(f"Smoothing failed: {e}")
            break
            
    return verts.tolist()


def _estimate_vertex_normals(vertices: np.ndarray) -> np.ndarray:
    """Estimate outward vertex normals for a convex polyhedron.

    Uses ConvexHull triangles and area-weighted face normals, then normalizes.
    """
    hull = ConvexHull(vertices)
    centroid = np.mean(vertices, axis=0)

    normals = np.zeros_like(vertices, dtype=float)
    for tri in hull.simplices:
        p0, p1, p2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        n = np.cross(p1 - p0, p2 - p0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        # Ensure outward direction
        tri_center = (p0 + p1 + p2) / 3.0
        if np.dot(n, tri_center - centroid) < 0:
            n = -n
        # Area-weighted accumulation
        normals[tri[0]] += n
        normals[tri[1]] += n
        normals[tri[2]] += n

    # Normalize
    lens = np.linalg.norm(normals, axis=1)
    lens[lens < 1e-12] = 1.0
    normals = normals / lens[:, None]
    return normals


def _coherent_noise_3d(x: float, y: float, z: float, seed: int = 42) -> float:
    """Coherent noise in [-1, 1].

    Prefers opensimplex if installed; otherwise falls back to a deterministic
    smooth-ish hash-based function.
    """
    try:
        import opensimplex

        # Cache per-seed generator to avoid re-init cost
        if not hasattr(_coherent_noise_3d, "_gens"):
            _coherent_noise_3d._gens = {}
        gens = _coherent_noise_3d._gens
        if seed not in gens:
            gens[seed] = opensimplex.OpenSimplex(seed=seed)
        return float(gens[seed].noise3(x=x, y=y, z=z))
    except Exception:
        # Fallback: smooth trigonometric mix (deterministic, continuous)
        s = float(seed) * 0.001
        v = (
            math.sin(0.07 * x + 1.3 * s)
            + math.sin(0.09 * y + 2.1 * s)
            + math.sin(0.05 * z + 0.7 * s)
        ) / 3.0
        # Map roughly into [-1, 1]
        return max(-1.0, min(1.0, v))

def perlin_noise_modification(vertices, amplitude):
    """
    使用 OpenSimplex 噪声修改多面体顶点位置。

    步骤一：对于每个缩放后的顶点 P。
    步骤二：计算从多面体质心指向顶点 P 的法向量 N（归一化向量）。
    步骤三：计算 OpenSimplex 噪声值 n = Noise(P_x, P_y, P_z)。
    步骤四：将顶点 P 沿法向量 N 移动 A * n 的距离，得到新位置 P'。

    参数:
    - vertices: 列表，包含多面体的顶点坐标，每个顶点为 (x, y, z) 元组。
    - amplitude: 浮点数，控制凹凸幅度的振幅参数 A。

    返回:
    - 修改后的顶点列表，每个顶点为 (x, y, z) 元组。

    异常:
    - 如果顶点列表为空，或振幅为 NaN，抛出 ValueError。
    """
    # 检查输入有效性
    if len(vertices) == 0:
        raise ValueError("顶点列表不能为空")
    if math.isnan(amplitude):
        raise ValueError("振幅参数不能为 NaN")

    verts = np.asarray(vertices, dtype=float)
    if len(verts) < 4:
        return vertices

    normals = _estimate_vertex_normals(verts)

    # 为了让噪声具有“尺度可控”，这里引入一个坐标缩放系数（类似论文里的相干噪声场）
    # 经验：以颗粒尺度(几十微米)而言，0.05~0.15 的频率较合适
    noise_freq = 0.08
    seed = 42

    modified = verts.copy()
    for i, p in enumerate(verts):
        n_value = _coherent_noise_3d(
            x=float(p[0]) * noise_freq,
            y=float(p[1]) * noise_freq,
            z=float(p[2]) * noise_freq,
            seed=seed,
        )
        modified[i] = p + float(amplitude) * float(n_value) * normals[i]

    modified_vertices = [tuple(v) for v in modified]

    # 步骤五：应用 Laplacian 平滑
    smoothed_vertices = laplacian_smoothing(modified_vertices, iterations=2, alpha=0.5)

    return smoothed_vertices


def generate_voronoi_perlin_particle(
    bounds=((0.0, 200.0), (0.0, 200.0), (0.0, 200.0)),
    expected_seed_count: int = 50,
    target_diameter_um: float | None = None,
    mu_ln: float = 4.0,
    sigma_ln: float = 0.3,
    diameter_range_um=(30.0, 90.0),
    amplitude_um: float = 2.5,
    random_seed: int = 42,
) -> list[tuple[float, float, float]]:
    """生成单个不规则颗粒（Voronoi–Perlin 复合模型）。

    与论文流程一致：
    1) 泊松点过程生成种子点并构造 3D Voronoi
    2) 选取一个有界闭单元作为骨架
    3) 按体积等效直径约束进行围绕质心缩放
    4) 沿外法线注入相干噪声并进行 Laplacian 平滑
    """
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds
    Lx, Ly, Lz = (x_max - x_min), (y_max - y_min), (z_max - z_min)
    volume = Lx * Ly * Lz
    if volume <= 0:
        raise ValueError("Invalid bounds")

    intensity = float(expected_seed_count) / float(volume)
    seeds = voronoi_generator.generate_poisson_seeds(bounds, intensity, random_seed=random_seed)
    polyhedrons = voronoi_generator.generate_voronoi_polyhedrons(seeds)
    if not polyhedrons:
        raise RuntimeError("Voronoi generation produced no bounded cells; try increasing seed count or bounds")

    # 选一个“相对居中”的单元：优先取种子点靠近域中心的
    center = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0])
    candidates = []
    for pid, verts in polyhedrons.items():
        p = np.array(seeds[pid])
        candidates.append((np.linalg.norm(p - center), pid))
    candidates.sort(key=lambda t: t[0])
    chosen_id = candidates[min(3, len(candidates) - 1)][1]  # 略偏中心，避免边界单元过扁
    verts = polyhedrons[chosen_id]

    # 目标粒径
    if target_diameter_um is None:
        target_diameter_um = size_distribution.sample_truncated_lognormal(
            mu_ln=mu_ln,
            sigma_ln=sigma_ln,
            size_range=diameter_range_um,
            rng=None,
        )

    # 围绕质心缩放到 D_eq
    verts = voronoi_generator.scale_polyhedron_to_eq_diameter(verts, float(target_diameter_um))

    # 噪声扰动 + 平滑
    verts = perlin_noise_modification(verts, float(amplitude_um))

    # 扰动会改变体积，为了严格满足 D_eq ∈ [30,90] 的硬约束，这里再次按 D_eq 进行缩放
    verts = voronoi_generator.scale_polyhedron_to_eq_diameter(verts, float(target_diameter_um))
    return [tuple(v) for v in np.asarray(verts, dtype=float)]


# 示例用法
# if __name__ == "__main__":
#
#
#     # 定义空间范围：x∈[0,10], y∈[0,5], z∈[-1,1]
#         space_bounds = ((0, 10), (0, 5), (-1, 1))
#
#         # 设置点密度（每单位体积10个点）
#         density = 10.0
#
#         possion_seeds=[]
#         seeds = voronoi_generator.generate_poisson_seeds(space_bounds, density, random_seed=42)
#         polyhedrons=voronoi_generator.generate_voronoi_polyhedrons(seeds)
#         for i in range(0, len(polyhedrons)):
#             possion_seeds.append( size_distribution.generate_normal_particle_size(
#                 60,
#                 10,
#             ))
#         vertices=voronoi_generator.scale_polyhedrons(polyhedrons,possion_seeds)
#         amplitude = 0.1  # 振幅参数 A
#
#     # 应用 OpenSimplex 噪声修改
#         modified = perlin_noise_modification(vertices[0], amplitude)
# #          # print("原始顶点:", vertices[0][:2], "...")  # 仅打印前两个示例
#         print("修改后顶点:", modified[:2], "...")  # 仅打印前两个示例