import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
sys.path.append('src')
import section_analysis
import spatial_analysis

# Use Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("======== 空间分布模式分析 (问题2) ========")
    
    # 1. Load data
    data_path = os.path.join('outputs', 'particles_data.pkl')
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return
        
    print(f"正在从 {data_path} 加载颗粒数据...")
    with open(data_path, 'rb') as f:
        particles_data = pickle.load(f)
        
    print(f"已加载 {len(particles_data)} 个颗粒。")
    
    # 2. Slice at specific height
    container_height = 20000.0 # From paper
    container_radius = 500.0   # From paper
    
    # Calculate packing height (approximate max Z)
    all_z = []
    for p in particles_data:
        all_z.extend(p[:, 2])
    max_z = np.max(all_z) if all_z else 0
    print(f"堆积体估算高度: {max_z:.2f} um")
    
    target_z = max_z / 2.0
    print(f"选定的裁切截面高度 z_0: {target_z:.2f} um")
    
    centroids = []
    
    print("正在提取截面质心...")
    for verts in particles_data: # tqdm(particles_data, desc="Slicing"):
        poly = section_analysis.get_slice_polygon(verts, target_z)
        if poly is not None:
             # Calculate centroid
             # Simple mean of vertices is rough centroid. 
             # For polygon area centroid (center of mass), it's more complex, 
             # but mean of vertices is usually sufficient for "location" in point process statistics
             # unless shapes are very non-convex or irregular. 
             # Given they are convex cuts of Voronoi cells, mean is very close to true centroid.
             c = np.mean(poly, axis=0)
             centroids.append(c)
             
    centroids = np.array(centroids)
    n_points = len(centroids)
    print(f"在截面 z={target_z:.2f} 处提取了 {n_points} 个颗粒质心。")
    
    if n_points < 10:
        print("警告: 点数量太少，无法进行有效的空间统计分析。")
        if n_points < 2: return
    
    # 3. Spatial Analysis (Ripley's K and L)
    print("\n开始 Ripley's K 函数分析...")
    
    # Define r range (0 to R/2 usually, or up to R)
    # Ripley's K is most reliable for r < domain_radius / 2
    max_r = container_radius / 1.5
    r_values = np.linspace(0, max_r, 50)
    # Remove 0 to avoid division by zero in some implementations (though handled in our func)
    r_values = r_values[1:] 
    
    # Calculate observed L
    k_obs = spatial_analysis.calculate_ripleys_k_circle(centroids, r_values, container_radius)
    l_obs = spatial_analysis.calculate_l_function(k_obs, r_values)
    
    # 4. Monte Carlo Simulation for Confidence Envelope
    # Using fewer simulations for speed in demo, e.g. 19 or 39 for rough envelope
    # Paper suggests more (999), but that takes time.
    n_sims = 40 
    print(f"正在运行蒙特卡洛模拟 (Simulations={n_sims}, Points={n_points})...")
    l_min, l_max, l_mean = spatial_analysis.monte_carlo_envelope(n_points, r_values, container_radius, n_sims)
    
    # 5. Plotting
    print("正在绘制分析结果...")
    plt.figure(figsize=(10, 8))
    
    # Plot L_csr = 0 reference
    plt.axhline(y=0, color='k', linestyle=':', label='CSR (完全空间随机)')
    
    # Plot Envelope
    plt.fill_between(r_values, l_min, l_max, color='gray', alpha=0.3, label=f'95% 置信区间 (N_sim={n_sims})')
    
    # Plot Observed
    plt.plot(r_values, l_obs, 'r-', linewidth=2, label='观测数据 L(r)')
    
    plt.title(f"Ripley's L-function 分析 (N={n_points} @ z={target_z:.1f}um)")
    plt.xlabel('距离 r (um)')
    plt.ylabel('L(r) = sqrt(K(r)/pi) - r')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Interpretation text
    text_str = "L(r) > 0: 聚集分布\nL(r) < 0: 均匀/规则分布"
    plt.text(0.05, 0.05, text_str, transform=plt.gca().transAxes, 
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    output_file = os.path.join('outputs', 'figure_3_ripleys_l.png')
    plt.savefig(output_file)
    print(f"结果已保存至 {output_file}")
    
    # Show scatter of points for verification
    plt.figure(figsize=(8, 8))
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(container_radius*np.cos(theta), container_radius*np.sin(theta), 'k--')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=10, c='b', alpha=0.6, label='颗粒质心')
    plt.axis('equal')
    plt.title(f"截面质心分布 (z={target_z:.1f}um)")
    plt.legend()
    plt.savefig(os.path.join('outputs', 'figure_3_centroids_map.png'))
    
    plt.show()

if __name__ == "__main__":
    main()
