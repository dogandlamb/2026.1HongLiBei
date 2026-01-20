import math
import sys

import size_distribution
import voronoi_generator


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
    if not vertices:
        raise ValueError("顶点列表不能为空")
    if math.isnan(amplitude):
        raise ValueError("振幅参数不能为 NaN")

    try:
        import opensimplex
    except ImportError:
        print("错误：需要 'opensimplex' 库来生成三维噪声。请通过命令 'pip install opensimplex' 安装。")
        sys.exit(1)

    # 初始化 OpenSimplex 噪声生成器（可指定 seed 固定噪声模式，如 opensimplex.OpenSimplex(seed=42)）
    noise_gen = opensimplex.OpenSimplex(seed=42)

    # 步骤二：计算多面体质心（所有顶点的平均位置）
    n_vertices = len(vertices)
    centroid_x = sum(v[0] for v in vertices) / n_vertices
    centroid_y = sum(v[1] for v in vertices) / n_vertices
    centroid_z = sum(v[2] for v in vertices) / n_vertices
    centroid = (centroid_x, centroid_y, centroid_z)

    # 初始化修改后的顶点列表
    modified_vertices = []

    # 步骤一：遍历每个顶点 P
    for p in vertices:
        p_x, p_y, p_z = p

        # 计算从质心指向 P 的向量
        vec_x = p_x - centroid[0]
        vec_y = p_y - centroid[1]
        vec_z = p_z - centroid[2]

        # 归一化向量以获得单位法向量 N
        magnitude = math.sqrt(vec_x ** 2 + vec_y ** 2 + vec_z ** 2)
        if magnitude == 0:
            # 边界情况：顶点与质心重合时不移动
            n_x, n_y, n_z = 0.0, 0.0, 0.0
        else:
            n_x = vec_x / magnitude
            n_y = vec_y / magnitude
            n_z = vec_z / magnitude

        # 步骤三：使用 OpenSimplex 计算三维噪声值（范围 [-1, 1]）
        # 注意：OpenSimplex 的 noise3 方法参数为 (x, y, z)，返回值范围 [-1, 1]
        n_value = noise_gen.noise3(x=p_x, y=p_y, z=p_z)

        # 步骤四：计算新位置 P' = P + A * n * N
        displacement_x = amplitude * n_value * n_x
        displacement_y = amplitude * n_value * n_y
        displacement_z = amplitude * n_value * n_z
        p_new = (p_x + displacement_x, p_y + displacement_y, p_z + displacement_z)

        modified_vertices.append(p_new)

    return modified_vertices


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