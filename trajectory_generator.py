import numpy as np
from scipy.stats import truncnorm, vonmises_fisher


def generate_3d_trajectory(N: int, mu_l: float, sigma_l: float, kappa: float, r0: float):
    """
    生成三维随机行走轨迹（基于Von Mises-Fisher方向分布和截断正态步长分布）

    参数:
        N (int): 轨迹点数量
        mu_l (float): 步长分布均值 (μm)
        sigma_l (float): 步长分布标准差 (μm)
        kappa (float): 方向分布集中度参数 (κ>0，值越大方向越集中)
        r0 (float): 初始核心半径 (μm)

    返回:
        trajectory (np.ndarray): 三维轨迹点数组，形状为(N+1, 3)
        step_lengths (np.ndarray): 各步长数组，形状为(N,)
    """
    # 初始化轨迹（从原点开始）
    trajectory = [np.array([0.0, 0.0, 0.0])]
    step_lengths = []

    # 截断正态分布参数（确保步长非负）
    a = (0 - mu_l) / sigma_l  # 左截断点
    b = np.inf  # 右截断点（无限制）

    # 调试信息容器
    debug_info = []

    for i in range(N):
        # --------------------------
        # 1. 方向向量生成（强化安全处理）
        # --------------------------
        if i == 0:
            # 初始方向：沿Z轴正方向（已归一化）
            mu_dir = np.array([0.0, 0.0, 1.0])
        else:
            # 计算前一步的方向向量
            prev_diff = trajectory[-1] - trajectory[-2]
            norm_val = np.linalg.norm(prev_diff)

            # 多重安全验证
            if norm_val < 1e-10 or np.any(np.isnan(prev_diff)) or np.any(np.isinf(prev_diff)):
                # 重新生成随机构造单位向量
                random_dir = np.random.randn(3)
                mu_dir = random_dir / np.linalg.norm(random_dir)

                # 记录调试信息
                debug_info.append({
                    'step': i,
                    'type': 'zero_or_invalid_vector',
                    'prev_point1': trajectory[-2],
                    'prev_point2': trajectory[-1],
                    'diff': prev_diff,
                    'norm': norm_val,
                    'new_dir': mu_dir
                })
            else:
                # 归一化得到单位方向向量
                mu_dir = prev_diff / norm_val

        # 精确验证单位向量（浮点容差1e-8）
        norm_before_call = np.linalg.norm(mu_dir)
        if not (0.99999 < norm_before_call < 1.00001):
            # 精确修正方向向量
            mu_dir = mu_dir / norm_before_call
            debug_info.append({
                'step': i,
                'type': 'normalization_adjustment',
                'norm_before': norm_before_call,
                'norm_after': np.linalg.norm(mu_dir)
            })

        # --------------------------
        # 2. 步长生成（截断正态分布）
        # --------------------------
        step_length = truncnorm.rvs(a, b, loc=mu_l, scale=sigma_l)
        step_lengths.append(step_length)

        # --------------------------
        # 3. 生成候选方向（Von Mises-Fisher分布）
        # --------------------------
        try:
            # 确保调用参数正确
            direction = vonmises_fisher.rvs(kappa, mu_dir)
        except Exception as e:
            # 详细记录错误信息
            debug_info.append({
                'step': i,
                'error': str(e),
                'mu_dir': mu_dir,
                'norm': np.linalg.norm(mu_dir),
                'kappa': kappa
            })
            # 安全回退：生成随机方向
            random_dir = np.random.randn(3)
            direction = random_dir / np.linalg.norm(random_dir)

        # --------------------------
        # 4. 更新轨迹点
        # --------------------------
        next_point = trajectory[-1] + direction * step_length
        trajectory.append(next_point)

    # 返回轨迹和调试信息
    return np.array(trajectory), np.array(step_length)
