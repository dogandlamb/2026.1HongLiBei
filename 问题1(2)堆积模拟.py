import os
import sys

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT_DIR, 'src'))

from packing_simulation import run_simulation, save_to_glb, visualize_packing, PackingParams

if __name__ == "__main__":
    # --- 参数配置区域 ---
    # 1. 显式设置颗粒数量 (原代码内部读取此环境变量)
    num_particles = 5000  # 建议先用少一点的颗粒数测试，比如 300
    os.environ['PACKING_N'] = str(num_particles)

    # 2. 显式定义模拟参数
    params = PackingParams()
    
    # [容器设置]
    params.container_radius_um = 500.0   # 容器半径 (微米)
    params.container_height_um = 20000.0 # 容器高度 (微米)
    
    # [颗粒生成设置]
    params.diameter_range_um = (30.0, 90.0) # 颗粒直径范围
    params.diameter_mu_ln = 4.0             # 对数正态分布 mu
    params.diameter_sigma_ln = 0.3          # 对数正态分布 sigma
    
    # [模拟迭代设置] - 调整此处可加速
    params.dt_s = 5e-7        # 时间步长: 增大可加速但可能导致不稳定 (推荐 1e-7 ~ 5e-7)
    params.max_steps = 10000   # 最大迭代步数
    params.cell_size_um = 120.0 # 空间网格大小 (略大于最大颗粒直径)

    print(f"开始模拟: 颗粒数={num_particles}, dt={params.dt_s}, 最大步数={params.max_steps}")
    
    final_particles, container = run_simulation(params)

    if os.environ.get("PACKING_EXPORT_GLB", "0") == "1":
        save_to_glb(final_particles, container, "outputs/output.glb")

    # 保存给问题2使用
    import pickle
    os.makedirs('outputs', exist_ok=True)
    particles_data = [p.get_world_vertices() for p in final_particles]
    with open('outputs/particles_data.pkl', 'wb') as f:
        pickle.dump(particles_data, f)

    visualize_packing(final_particles, container)
