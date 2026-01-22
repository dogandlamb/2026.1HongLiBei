import math
import random
from typing import Optional, Tuple


def sample_truncated_lognormal(
    mu_ln: float = 4.0,
    sigma_ln: float = 0.3,
    size_range: Tuple[float, float] = (30.0, 90.0),
    max_attempts: int = 10_000,
    rng: Optional[random.Random] = None,

) -> float:
    r"""从截断对数正态分布采样粒径（单位：μm）。

    与论文一致：$\ln(D) \sim \mathcal{N}(\mu, \sigma^2)$，并截断到 $D\in[a,b]$。

    参数:
    - mu_ln, sigma_ln: 对数空间参数（对应论文中的 μ, σ）
    - size_range: (a, b)，单位 μm
    - rng: 可传入 random.Random 以便复现实验
    """
    low, high = size_range
    if low >= high:
        raise ValueError("Invalid size_range: low must be < high")
    if sigma_ln <= 0:
        raise ValueError("sigma_ln must be positive")

    if rng is None:
        rng = random

    for _ in range(max_attempts):
        # Box-Muller 生成标准正态
        u1, u2 = rng.random(), rng.random()
        z = math.sqrt(-2.0 * math.log(max(u1, 1e-12))) * math.cos(2 * math.pi * u2)
        d = math.exp(mu_ln + z * sigma_ln)
        if low <= d <= high:
            return round(d, 2)

    raise RuntimeError(f"生成失败: 超过 {max_attempts} 次仍未落入范围 {size_range}")


def generate_normal_particle_size(
    mu: float,
    sigma: float,
    size_range: Tuple[float, float] = (30, 90),
    max_attempts: int = 1000,
    debug: bool = False,
) -> float:
    """兼容旧接口。

    注意：历史上该函数名不准确——实际生成的是“对数正态(LogNormal)”粒径，
    且 mu/sigma 为 ln(D) 的参数。
    """
    if debug:
        print(
            f"DEBUG: LogNormal(mu_ln={mu}, sigma_ln={sigma}), range={size_range}, max={max_attempts}"
        )
    return sample_truncated_lognormal(mu_ln=mu, sigma_ln=sigma, size_range=size_range, max_attempts=max_attempts)

# for i in range(1, 6):
#         print(f"\n=== 样本 #{i} ===")
#         size = generate_normal_particle_size(
#             mu=60,
#             sigma=10,
#             debug=True  # 开启调试模式
#         )
#         print(f"生成结果: {size}μm")

