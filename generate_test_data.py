import sys
import os
import numpy as np
import pickle
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 添加 src 到路径以使用现有的生成逻辑
sys.path.append('src')
import voronoi_generator
import size_distribution
import particle_generator

def visualize_generated_data(particles_data, radius, height):
    print("正在可视化生成的颗粒...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制容器轮廓
    z = np.linspace(0, height, 2)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.1, color='gray')
    ax.plot(x_grid[0], y_grid[0], z_grid[0], color='k') # Bottom
    
    # 绘制颗粒
    # 为了性能，如果颗粒太多，只画一部分
    to_draw = particles_data if len(particles_data) < 200 else particles_data[:200]
    
    colors = plt.cm.jet(np.linspace(0, 1, len(to_draw)))
    
    for i, verts in enumerate(to_draw):
        try:
            hull = ConvexHull(verts)
            faces = [verts[s] for s in hull.simplices]
            poly = Poly3DCollection(faces, alpha=0.8, edgecolor='k', facecolor=colors[i], linewidths=0.2)
            ax.add_collection3d(poly)
        except:
            pass

    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_zlabel('Z (um)')
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(0, height)
    ax.set_box_aspect([1, 1, height/radius/2]) # Aspect ratio
    ax.set_title(f"Generated Test Particles ({len(particles_data)} count)")
    
    plt.show()

def generate_fast_test_data(num_particles=100, container_radius=500.0, container_height=1000.0):
    print(f"正在快速生成 {num_particles} 个测试颗粒数据...")
    
    # 1. 生成 Voronoi 种子和多面体
    # 为了保证有点随机性，生成稍多一点的种子
    bounds = ((0, 100), (0, 100), (0, 100)) # 基础生成空间
    # 密度设置大一点以生成足够多的单元
    density = num_particles / (100*100*100) * 2 
    seeds = voronoi_generator.generate_poisson_seeds(bounds, max(density, 1e-4), random_seed=42)
    
    # 如果种子不够，强制补足 (简单随机)
    if len(seeds) < num_particles:
        extra = np.random.rand(num_particles - len(seeds), 3) * 100
        seeds.extend(extra.tolist())
        
    polyhedrons = voronoi_generator.generate_voronoi_polyhedrons(seeds)
    keys = list(polyhedrons.keys())
    
    if not keys:
        print("错误：无法生成 Voronoi 多面体")
        return

    particles_data = []
    
    # 2. 处理每个颗粒
    count = 0
    import random
    
    while count < num_particles:
        # 随机取一个形状模版
        k = random.choice(keys)
        verts = np.array(polyhedrons[k])
        
        # 计算体积用于缩放
        try:
            hull = ConvexHull(verts)
            vol = hull.volume
        except:
            continue
            
        # 目标粒径：对数正态分布
        target_d = size_distribution.generate_normal_particle_size(4.0, 0.3, (30, 90))
        current_d = (6 * vol / np.pi)**(1/3)
        if current_d == 0: continue
        
        scale_factor = target_d / current_d
        
        # 缩放
        verts = verts * scale_factor
        
        # 施加噪声 (模拟真实形状)
        # 将 numpy 转为 list 以适配函数接口，然后再转回
        verts = particle_generator.perlin_noise_modification(verts.tolist(), 3.0)
        verts = np.array(verts)

        # 3. 随机放置在容器内 (简单的堆积模拟替代方案)
        # 对于测试截面分析，我们不需要物理上严密的堆积，
        #只需要颗粒在空间中有分布即可。
        # 我们将它们随机分布在圆柱体内。
        
        # 随机高度 Z
        z = random.uniform(0, container_height * 0.8)
        
        # 随机水平位置 (极坐标)
        r = random.uniform(0, container_radius - target_d/2)
        theta = random.uniform(0, 2*np.pi)
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # 随机旋转
        # 简单起见，这里不旋转或者施加一个随机旋转矩阵
        # 随机欧拉角
        angles = np.random.rand(3) * 2 * np.pi
        # 生成旋转矩阵 (简化版: 绕Z, Y, X)
        Rx = np.array([[1,0,0],[0,np.cos(angles[0]),-np.sin(angles[0])],[0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],[0,1,0],[-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],[np.sin(angles[2]),np.cos(angles[2]),0],[0,0,1]])
        R = Rz @ Ry @ Rx
        
        # 中心化并旋转
        center = np.mean(verts, axis=0)
        verts_centered = verts - center
        verts_rotated = np.dot(verts_centered, R.T)
        
        # 移至目标位置
        final_verts = verts_rotated + np.array([x, y, z])
        
        particles_data.append(final_verts)
        count += 1
        
    print(f"成功生成 {len(particles_data)} 个颗粒的数据。")
    
    # 4. 保存为 pickle
    output_dir = 'outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, 'particles_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(particles_data, f)
        
    print(f"数据已保存至: {output_path}")
    return particles_data

if __name__ == "__main__":
    # 生成 200 个颗粒用于测试
    data = generate_fast_test_data(num_particles=10000)
    visualize_generated_data(data, 500.0, 1000.0)
