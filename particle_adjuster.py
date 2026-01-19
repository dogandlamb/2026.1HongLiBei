import numpy as np
import trimesh
from scipy.spatial import ConvexHull


def adjust_particle_size(mesh, target_dv_min=30, target_dv_max=90):
    """
    调整颗粒尺寸并验证形态特征

    参数:
        mesh (trimesh.Trimesh): 输入网格
        target_dv_min (float): 最小目标等效直径(μm)
        target_dv_max (float): 最大目标等效直径(μm)

    返回:
        trimesh.Trimesh: 调整后的网格
        dict: 形态特征指标
    """
    # 计算当前体积和等效直径
    current_volume = mesh.volume
    current_dv = (6 * current_volume / np.pi) ** (1 / 3)

    # 随机选择目标等效直径
    target_dv = np.random.uniform(target_dv_min, target_dv_max)
    target_volume = (np.pi / 6) * target_dv ** 3

    # 计算缩放因子并应用
    scale_factor = (target_volume / current_volume) ** (1 / 3)
    scaled_mesh = mesh.apply_scale(scale_factor)

    # 计算形态指标
    metrics = calculate_morphology_metrics(scaled_mesh)

    return scaled_mesh, metrics


def calculate_morphology_metrics(mesh):
    """计算颗粒形态特征指标"""
    # 计算表面积
    surface_area = mesh.area

    # 计算凸包
    convex_hull = ConvexHull(mesh.vertices)
    hull_volume = convex_hull.volume

    # 计算形态指标
    volume = mesh.volume
    equivalent_diameter = (6 * volume / np.pi) ** (1 / 3)

    return {
        'equivalent_diameter': equivalent_diameter,  # 等效直径(μm)
        'sphericity': (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / surface_area,  # 球形度
        'convexity': volume / hull_volume,  # 凸度
        'surface_area': surface_area,  # 表面积(μm²)
        'volume': volume  # 体积(μm³)
    }
