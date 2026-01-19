from trajectory_generator import generate_3d_trajectory
from mesh_constructor import construct_mesh_from_trajectory
from particle_adjuster import adjust_particle_size



# === 步骤1：轨迹生成参数 ===
trajectory_params = {
    'N': 1500,        # 轨迹点数
    'mu_l': 55.0,      # 平均步长(μm)
    'sigma_l': 3.5,   # 步长标准差
    'kappa': 0.5,     # 方向集中度
    'r0': 15.0        # 初始核心半径(μm)
}

# === 步骤2：尺寸调整参数 ===
adjustment_params = {
    'target_dv_min': 30,  # 最小目标粒径(μm)
    'target_dv_max': 90   # 最大目标粒径(μm)
}

# 1. 生成随机轨迹（仅传递轨迹参数）
trajectory, step_lengths = generate_3d_trajectory(**trajectory_params)

# 2. 构建三维网格
particle_mesh = construct_mesh_from_trajectory(trajectory, step_lengths)

# 3. 调整尺寸并验证（传递调整参数）
adjusted_mesh, metrics = adjust_particle_size(
    particle_mesh,
    **adjustment_params  #  正确传递尺寸参数
)

# 4. 输出结果
print(f"等效直径: {metrics['equivalent_diameter']:.2f} μm")
print(f"球形度: {metrics['sphericity']:.4f}")
print(f"凸度: {metrics['convexity']:.4f}")

#可视化

adjusted_mesh.show()