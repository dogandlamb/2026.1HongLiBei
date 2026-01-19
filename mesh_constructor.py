import numpy as np
import trimesh
from scipy.spatial import KDTree


def construct_mesh_from_trajectory(trajectory, step_lengths):
    """
    从三维轨迹构建粒子网格（扫掠算法）

    参数:
        trajectory (np.ndarray): 轨迹点数组，形状为(N, 3)
        step_lengths (np.ndarray, list, float): 步长信息
    """
    # 确保轨迹为NumPy数组
    if not isinstance(trajectory, np.ndarray):
        trajectory = np.array(trajectory, dtype=np.float64)

    num_points = trajectory.shape[0]

    # ============= 修改部分：解决len()边界问题 =============
    # 更健壮的步长处理（避免使用len()）
    if isinstance(step_lengths, (int, float)):
        # 单个数值：所有步长相同
        step_array = np.full(num_points - 1, float(step_lengths))
    else:
        # 处理数组/列表类型
        step_array = np.asarray(step_lengths, dtype=np.float64).flatten()

        # 使用size属性替代len()
        if step_array.size == 1:
            # 单个元素：广播到所有步长
            step_array = np.full(num_points - 1, float(step_array[0]))
        elif step_array.size != num_points - 1:
            raise ValueError(
                f"步长数量({step_array.size})应与轨迹点数-1({num_points - 1})匹配"
            )
    # ============= 修改结束 =============

    # 计算每个轨迹点的半径
    radii = np.zeros(num_points)
    radii[0] = 0.5 * step_array[0]
    radii[-1] = 0.5 * step_array[-1]

    for i in range(1, num_points - 1):
        radii[i] = 0.5 * (0.7 * step_array[i - 1] + 0.3 * step_array[i])

    # 生成截面圆
    tangents = np.zeros_like(trajectory)
    tangents[0] = trajectory[1] - trajectory[0]
    tangents[0] /= np.linalg.norm(tangents[0])

    tangents[-1] = trajectory[-1] - trajectory[-2]
    tangents[-1] /= np.linalg.norm(tangents[-1])

    for i in range(1, num_points - 1):
        tangents[i] = trajectory[i + 1] - trajectory[i - 1]
        tangents[i] /= np.linalg.norm(tangents[i])

    # 生成圆环点集
    circle_points = []
    resolution = 16  # 圆的分辨率

    for i in range(num_points):
        # 创建正交基
        base_vec = np.array([1.0, 0.0, 0.0]) if abs(tangents[i][2]) > 0.9 else np.array([0.0, 0.0, 1.0])
        vec1 = base_vec - np.dot(base_vec, tangents[i]) * tangents[i]
        vec1 /= np.linalg.norm(vec1)
        vec2 = np.cross(tangents[i], vec1)

        # 生成圆上的点
        theta = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
        circle = trajectory[i] + radii[i] * (
                np.outer(np.cos(theta), vec1) +
                np.outer(np.sin(theta), vec2)
        )
        circle_points.append(circle)

    # 创建网格表面
    vertices = np.vstack(circle_points)
    faces = []

    # 连接相邻圆环
    for i in range(num_points - 1):
        for j in range(resolution):
            j_next = (j + 1) % resolution
            v0 = i * resolution + j
            v1 = i * resolution + j_next
            v2 = (i + 1) * resolution + j
            v3 = (i + 1) * resolution + j_next

            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    # 添加端盖
    # 起点封顶
    start_center = len(vertices)
    vertices = np.vstack([vertices, trajectory[0]])
    for j in range(resolution):
        j_next = (j + 1) % resolution
        faces.append([start_center, j, j_next])

    # 终点封底
    end_center = len(vertices)
    vertices = np.vstack([vertices, trajectory[-1]])
    last_circle_start = (num_points - 1) * resolution
    for j in range(resolution):
        j_next = (j + 1) % resolution
        idx = last_circle_start + j
        idx_next = last_circle_start + j_next
        faces.append([end_center, idx_next, idx])

    return trimesh.Trimesh(vertices=vertices, faces=faces)