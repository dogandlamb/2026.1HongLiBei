import sys
sys.path.append('src')
import size_distribution
import voronoi_generator
import particle_generator
import particle_analysis

from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
import trimesh

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_polyhedrons(polyhedrons_dict):
    # 创建一个大图，包含 2x5 = 10 个子图
    fig = plt.figure(figsize=(20, 10))
    
    # 获取前10个颗粒的键
    keys = list(polyhedrons_dict.keys())[:10]
    
    # 为不同多面体生成随机颜色
    colors = plt.cm.jet(np.linspace(0, 1, len(keys)))

    for i, key in enumerate(keys):
        vertices = polyhedrons_dict[key]
        
        # 创建子图
        ax = fig.add_subplot(2, 5, i + 1, projection='3d')
        
        # 将顶点列表转换为NumPy数组
        verts = np.array(vertices)

        # 计算几何参数
        params = particle_analysis.analyze_particle(verts)

        # 创建多面体对象（凸包）
        hull = ConvexHull(verts)

        # 绘制多面体表面
        faces = []
        for simplex in hull.simplices:
            faces.append(verts[simplex])

        poly = Poly3DCollection(faces,
                                alpha=0.7,
                                edgecolor='k',
                                facecolor=colors[i])
        ax.add_collection3d(poly)

        # 设置标题显示参数
        title_str = (f'ID:{key}\n'
                     f'D_eq:{params["Equivalent Diameter"]:.2f} Sp:{params["Sphericity"]:.2f}\n'
                     f'Flat:{params["Flatness"]:.2f} Elong:{params["Elongation"]:.2f}\n'
                     f'Rnd:{params["Roundness (Approx)"]:.2f}')
        ax.set_title(title_str, fontsize=8)

        # 自动调整坐标轴范围
        ax.set_xlim(verts[:,0].min(), verts[:,0].max())
        ax.set_ylim(verts[:,1].min(), verts[:,1].max())
        ax.set_zlim(verts[:,2].min(), verts[:,2].max())
        
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 保持比例
        ax.set_box_aspect([1, 1, 1])

        # 自动调整视角
        ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    # 默认工具栏通常就支持3D旋转（鼠标左键拖动）
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
                4.0,
                0.3,
            ))
        vertices=voronoi_generator.scale_polyhedrons(polyhedrons,possion_seeds)
        
        # 只显示前10个颗粒
        vertices = dict(list(vertices.items())[:10])
        
        amplitude = 2.5  # 振幅参数 A（论文标定最优值）

    # 应用 OpenSimplex 噪声修改
        for i in vertices:
            vertices[i] = particle_generator.perlin_noise_modification(vertices[i], amplitude)
        # 执行可视化
        plot_polyhedrons(vertices)