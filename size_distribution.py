import math
import random
from typing import Tuple


def generate_normal_particle_size(
        mu: float,
        sigma: float,
        size_range: Tuple[float, float] = (30, 90),
        max_attempts: int = 1000,
        debug: bool = False  # 新增调试模式参数
) -> float:
    """
    生成符合正态分布的随机粒径（单位：μm）

    参数:
    mu -- 目标正态分布的均值
    sigma -- 目标正态分布的标准差
    size_range -- 粒径允许的范围 (默认值: (30, 90))
    max_attempts -- 最大尝试次数 (防止无限循环)
    debug -- 调试模式开关 (默认关闭)

    返回:
    落在指定范围内的随机粒径值

    异常:
    ValueError -- 当无法生成有效值或输入参数无效时
    """
    # 验证参数有效性
    low, high = size_range
    if low >= high:
        raise ValueError("Invalid size range")
    if sigma <= 0:
        raise ValueError("Sigma must be positive")

    # Box-Müller变换生成标准正态分布
    def _box_muller() -> float:
        u1, u2 = random.random(), random.random()
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return z0

    # 调试信息：显示输入参数
    if debug:
        print(f"DEBUG: 生成参数 - 均值={mu}μm, 标准差={sigma}μm, 取值范围=[{low}, {high}]μm")

    # 通过逆变换法转换到目标正态分布
    for attempt in range(1, max_attempts + 1):
        # 生成标准正态随机数
        z = _box_muller()

        # 转换到目标分布
        particle_size = mu + z * sigma

        # 调试信息：显示每次尝试结果
        if debug:
            status = "符合要求" if low <= particle_size <= high else "超出范围"
            print(f"DEBUG: 尝试#{attempt} - 生成值={particle_size:.2f}μm ({status})")

        # 检查范围约束
        if low <= particle_size <= high:
            final_size = round(particle_size, 2)  # 保留两位小数
            if debug:
                print(f"DEBUG: 成功生成粒径 - {final_size}μm (总尝试次数: {attempt})")
            return final_size

    # 超过最大尝试次数
    err_msg = f"生成失败: 经过 {max_attempts} 次尝试仍无法生成符合要求的粒径"
    if debug:
        print(f"DEBUG: {err_msg}")
    raise RuntimeError(err_msg)

# for i in range(1, 6):
#         print(f"\n=== 样本 #{i} ===")
#         size = generate_normal_particle_size(
#             mu=60,
#             sigma=10,
#             debug=True  # 开启调试模式
#         )
#         print(f"生成结果: {size}μm")

