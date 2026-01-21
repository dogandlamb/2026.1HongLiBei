# import numpy as np
# from typing import List, Tuple
# import numpy as np
from scipy.spatial import Voronoi, ConvexHull
# from typing import List, Tuple, Dict
import numpy as np
# from scipy.spatial import ConvexHull
from typing import Dict, List, Tuple
import math



def generate_poisson_seeds(
        bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        intensity: float,
        random_seed: int = None
) -> List[Tuple[float, float, float]]:
    """
    在三维空间中生成符合 Poisson 点过程的种子点

    参数:
    bounds -- 三维空间边界 ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    intensity -- 点密度（单位体积内的平均点数）
    random_seed -- 随机种子（可选）

    返回:
    种子点坐标列表 [(x1, y1, z1), (x2, y2, z2), ...]
    """
    # 设置随机种子
    if random_seed is not None:
        np.random.seed(random_seed)

    # 解包空间边界
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds

    # 计算空间体积
    volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

    # 生成符合泊松分布的点数
    mean_points = intensity * volume
    num_points = np.random.poisson(mean_points)

    # 在三维空间内均匀生成点坐标
    x_coords = np.random.uniform(x_min, x_max, num_points)
    y_coords = np.random.uniform(y_min, y_max, num_points)
    z_coords = np.random.uniform(z_min, z_max, num_points)

    # 组合成坐标列表
    seed_points = list(zip(x_coords, y_coords, z_coords))

    return seed_points


# # 使用示例
# if __name__ == "__main__":
#     # 定义空间范围：x∈[0,10], y∈[0,5], z∈[-1,1]
#     space_bounds = ((0, 10), (0, 5), (-1, 1))
#
#     # 设置点密度（每单位体积10个点）
#     density = 10.0
#
#     # 生成种子点（固定随机种子确保可重复性）
#     seeds = generate_poisson_seeds(space_bounds, density, random_seed=42)
#
#     # 输出前5个点验证
#     print(f"生成种子点数量: {len(seeds)}")
#     print("前5个种子点坐标:")
#     for i, point in enumerate(seeds[:5]):
#         print(f"点{i + 1}: ({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f})")




def generate_voronoi_polyhedrons(
        seeds: List[Tuple[float, float, float]],
        compute_convex_hull: bool = True
) -> Dict[int, List[Tuple[float, float, float]]]:
    """
    根据种子点生成三维 Voronoi 多面体

    参数:
    seeds -- 种子点坐标列表，格式为 [(x1, y1, z1), (x2, y2, z2), ...]
    compute_convex_hull -- 是否计算凸包顶点（默认为True）

    返回:
    字典 {种子点索引: 对应多面体顶点坐标列表}
    """
    points = np.array(seeds)

    # 计算三维 Voronoi 图
    vor = Voronoi(points)

    # 存储结果：{种子索引: 顶点列表}
    polyhedrons = {}

    # 处理每个种子点对应的区域
    for idx, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]

        # 跳过无效区域（空区域或包含-1的区域）
        if not region or -1 in region:
            continue

        # 获取区域顶点坐标
        vertices = [tuple(vor.vertices[i]) for i in region]

        # 计算凸包顶点（确保多面体是凸的）
        if compute_convex_hull and len(vertices) > 3:
            hull = ConvexHull(vertices)
            hull_vertices = [tuple(vertices[i]) for i in hull.vertices]
            polyhedrons[idx] = hull_vertices
        else:
            polyhedrons[idx] = vertices

    return polyhedrons


# 示例使用
# if __name__ == "__main__":
#     # 生成种子点（使用之前实现的函数）
#
#
#     # 定义空间范围
#     space_bounds = ((0, 10), (0, 5), (-1, 1))
#     density = 5.0
#
#     # 生成种子点
#     seeds = generate_poisson_seeds(space_bounds, density, random_seed=42)
#     print(f"生成种子点数量: {len(seeds)}")
#
#     # 生成 Voronoi 多面体
#     polyhedrons = generate_voronoi_polyhedrons(seeds)
#
#     # 输出结果
#     print(f"\n生成多面体数量: {len(polyhedrons)}")
#     for idx, vertices in list(polyhedrons.items())[:3]:  # 打印前3个多面体
#         seed_point = seeds[idx]
#         print(f"\n种子点 {idx} @ ({seed_point[0]:.2f}, {seed_point[1]:.2f}, {seed_point[2]:.2f})")
#         print(f"多面体顶点数: {len(vertices)}")
#         for i, v in enumerate(vertices[:5]):  # 打印前5个顶点
#             print(f"  顶点 {i + 1}: ({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})")
#         if len(vertices) > 5:
#             print(f"  省略 {len(vertices) - 5} 个顶点...")




def scale_polyhedrons(
        polyhedrons: Dict[int, List[Tuple[float, float, float]]],
        target_diameters: List[float],
        verbose: bool = True
) -> Dict[int, List[Tuple[float, float, float]]]:
    """
    缩放多面体使其等效直径符合目标粒径分布

    参数:
    polyhedrons -- 原始多面体字典 {id: 顶点列表}
    target_diameters -- 目标粒径列表 (与多面体一一对应)
    verbose -- 是否打印缩放信息 (默认为True)

    返回:
    缩放后的多面体字典 {id: 新顶点列表}
    """
    # 验证输入
    if len(polyhedrons) != len(target_diameters):
        raise ValueError("多面体数量与目标粒径数量不一致")

    scaled_polyhedrons = {}
    scaling_factors = []
    original_diameters = []

    # 遍历每个多面体
    for idx, (poly_id, vertices) in enumerate(polyhedrons.items()):
        # 转换为numpy数组以便计算
        vertices_arr = np.array(vertices)

        # 计算凸包体积 (等效球体积)
        hull = ConvexHull(vertices_arr)
        volume = hull.volume

        # 计算当前等效直径
        d_v = (6 * volume / math.pi) ** (1 / 3)
        original_diameters.append(d_v)

        # 获取目标粒径
        d_target = target_diameters[idx]

        # 计算缩放因子
        s = d_target / d_v
        scaling_factors.append(s)

        # 缩放所有顶点
        scaled_vertices = [(x * s, y * s, z * s) for (x, y, z) in vertices]
        scaled_polyhedrons[poly_id] = scaled_vertices

        # 打印缩放信息
        if verbose:
            print(f"多面体 {poly_id}:")
            print(f"  原始等效直径: {d_v:.4f} μm, 目标直径: {d_target:.4f} μm")
            print(f"  缩放因子: {s:.4f}, 顶点数: {len(vertices)}")

    # 打印统计信息
    if verbose and original_diameters:
        avg_original = np.mean(original_diameters)
        avg_target = np.mean(target_diameters)
        avg_scaling = np.mean(scaling_factors)

        print("\n缩放统计:")
        print(f"  平均原始等效直径: {avg_original:.4f} μm")
        print(f"  平均目标直径: {avg_target:.4f} μm")
        print(f"  平均缩放因子: {avg_scaling:.4f}")
        print(f"  最小缩放因子: {min(scaling_factors):.4f}")
        print(f"  最大缩放因子: {max(scaling_factors):.4f}")

    return scaled_polyhedrons


# # 使用示例
# if __name__ == "__main__":
#     # 示例数据 - 实际应用中应使用真实生成的多面体和粒径
#     # 创建一个简单的四面体作为示例
#     tetrahedron = {
#         0: [
#             (1, 1, 1),
#             (1, -1, -1),
#             (-1, 1, -1),
#             (-1, -1, 1)
#         ]
#     }
#
#     # 目标粒径 (示例)
#     target_diameters = [50.0]  # 50μm
#
#     # 缩放多面体
#     scaled_poly = scale_polyhedrons(tetrahedron, target_diameters)
#
#     # 输出结果
#     print("\n缩放后的四面体顶点:")
#     for vertex in scaled_poly[0]:
#         print(f"  ({vertex[0]:.4f}, {vertex[1]:.4f}, {vertex[2]:.4f})")
