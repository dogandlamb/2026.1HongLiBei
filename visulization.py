import size_distribution
import voronoi_generator
import particle_generator

from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_polyhedrons(polyhedrons_dict):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 为不同多面体生成随机颜色
    colors = plt.cm.jet(np.linspace(0, 1, len(polyhedrons_dict)))

    for idx, (key, vertices) in enumerate(polyhedrons_dict.items()):
        # 将顶点列表转换为NumPy数组
        verts = np.array(vertices)

        # 创建多面体对象（凸包）
        hull = ConvexHull(verts)

        # 绘制多面体表面
        faces = []
        for simplex in hull.simplices:
            faces.append(verts[simplex])

        poly = Poly3DCollection(faces,
                                alpha=0.7,
                                edgecolor='k',
                                facecolor=colors[idx])
        ax.add_collection3d(poly)

        # 添加索引标签
        centroid = np.mean(verts, axis=0)
        ax.text(centroid[0], centroid[1], centroid[2],
                f'ID:{key}', fontsize=9, ha='center')

    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Polyhedron Visualization')

    # 自动调整视角
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    # 在plt.show()前添加这几行
    ax.set_box_aspect([1, 1, 1])  # 保持比例不变形
    plt.tight_layout()

    # 启用高级交互模式（关键！）
    plt.rcParams['toolbar'] = 'toolmanager'  # 激活工具栏
    fig.canvas.manager.toolmanager.add_tool('Rotate', Axes3D._rotate)  # 添加旋转工具
    plt.show()


# 示例数据（四面体 + 立方体）
# data = {
#     1: [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],  # 四面体
#     2: [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),  # 立方体
#         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
# }



# 示例用法



if __name__ == "__main__":


    # 定义空间范围：x∈[0,10], y∈[0,5], z∈[-1,1]
        space_bounds = ((0, 10), (0, 5), (-1, 1))

        # 设置点密度（每单位体积10个点）
        density = 10.0

        possion_seeds=[]
        seeds = voronoi_generator.generate_poisson_seeds(space_bounds, density, random_seed=42)
        polyhedrons=voronoi_generator.generate_voronoi_polyhedrons(seeds)
        for i in range(0, len(polyhedrons)):
            possion_seeds.append( size_distribution.generate_normal_particle_size(
                60,
                10,
            ))
        vertices=voronoi_generator.scale_polyhedrons(polyhedrons,possion_seeds)
        amplitude = 0.1  # 振幅参数 A

    # 应用 OpenSimplex 噪声修改
        for i in range(0, 9):
            vertices[i] = particle_generator.perlin_noise_modification(vertices[i], amplitude)
        # 执行可视化
        plot_polyhedrons(vertices)