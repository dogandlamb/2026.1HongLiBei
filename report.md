# B题：不规则颗粒三维随机堆积建模与截面微观结构表征（可复现代码版）

> 适用：哈尔滨工业大学（深圳）数学建模竞赛 B 题。
>
> 本报告配套代码：`run_demo.py` 与 `src/`。

## 1 问题理解与建模目标

- 圆柱容器内径 $D_c=1000\,\mu m$，半径 $R_c=500\,\mu m$。
- 颗粒等效粒径 $d\in[30,90]\,\mu m$，颗粒显著不规则（非球形、棱角/凹凸）。
- 需求：
  - **问题1**：构建单颗粒三维几何模型 + 圆柱内随机堆积模拟，并可视化三维填充。
  - **问题2**：任意高度横截面提取二维“颗粒截面”闭合区域，统计：
    1) 面积/等效直径等几何指标的分布拟合与检验；
    2) 截面质心点的空间分布模式（随机/团簇/均匀）量化。

## 2 问题1：三维随机堆积模型

### 2.1 单颗粒几何模型（不规则性）

我们用“**球面径向扰动**”构造一个可控的不规则闭合曲面。

- 先取等效直径 $d$，基准半径 $r_0=d/2$。
- 令单位球面方向向量为 $\mathbf u(\theta,\phi)$。
- 定义径向函数（多峰扰动）：

$$
 r(\mathbf u)=r_0\left(1+\sum_{k=1}^{K} a_k\cos\big( f_k\,\alpha(\mathbf u,\mathbf d_k)+\varphi_k\big)\right)
$$

其中：
- $K$ 为“凸起/凹陷”数量（代码里 `n_bumps`），
- $\mathbf d_k$ 为随机方向，
- $a_k$ 为扰动幅度（如 $0.04\sim0.14$），
- $f_k$ 为低阶频率（1–3），
- $\alpha(\mathbf u,\mathbf d_k)=\arccos(\mathbf u\cdot\mathbf d_k)$。

最后顶点坐标取 $\mathbf x=r(\mathbf u)\mathbf u$，即可得到不规则颗粒网格（`trimesh` icosphere 变形）。

**优点**：
- 封闭、可做截面、易随机化；
- 能体现凹凸/棱角的统计不规则性；
- 用少量参数控制“粗糙度”。

对应代码：[src/particle.py](src/particle.py)

### 2.2 三维堆积模型（沉降式随机堆积）

严格的离散元（DEM）需要求解接触力、摩擦、转动等，成本较高。竞赛建模中可采用**序贯沉降-接受拒绝**的启发式模型：

1. 逐个生成颗粒（尺寸+随机姿态）。
2. 从当前堆积顶部上方投放，执行若干步“向下位移 + 小随机横向漂移”。
3. 每步检查：
   - 是否仍在圆柱内（中心到轴线距离 $\le R_c - r_b$）；
   - 是否与已有颗粒发生重叠（用包围球快速剪枝 + 网格碰撞判定）。
4. 一旦继续下落导致碰撞，则将上一可行位置视为“沉降稳定”并接受该颗粒。

该模型能产生：
- 边界效应（靠近壁面形态受限）；
- 随机堆积微结构；
- 合理的孔隙与互锁现象（统计层面足够用于截面分析）。

对应代码：[src/packing.py](src/packing.py)

### 2.3 可视化与导出

- 导出 `packing.glb`：可用 Windows 自带 3D 查看器或 Blender 打开。
- 输出 `topdown.png`：俯视粒子质心分布。

对应代码：[src/viz.py](src/viz.py)

## 3 问题2：截面提取与统计表征

### 3.1 横截面提取（二维颗粒截面）

对高度 $z=z_0$ 的平面进行截切：
- 对每个颗粒网格与平面求交得到闭合曲线（多条回路可能出现）。
- 将交线投影到 XY 平面得到 2D 多边形。
- 与容器圆形窗口求交裁剪，得到截面可见区域。

对应代码：[src/section.py](src/section.py)

### 3.2 截面几何属性统计与分布拟合

对每个颗粒截面多边形计算：
- 面积 $A$；
- 周长 $P$；
- 圆度（紧致度）

$$
C=\frac{4\pi A}{P^2}\in(0,1]
$$

- 面积等效直径（2D）

$$
 d_{eq}=\sqrt{\frac{4A}{\pi}}
$$

然后对 $A$ 或 $d_{eq}$ 的样本进行候选分布拟合（对数正态、韦布尔、Gamma），并用 Kolmogorov–Smirnov (KS) 检验比较优劣（以 $p$ 值高者为优）。

对应代码：[src/analysis.py](src/analysis.py)

### 3.3 空间位置分布规律（点过程指标）

将截面上每个颗粒截面的质心视为点集 $\{\mathbf x_i\}$，给出 3 个指标：

1) 最近邻距离均值 $\bar r$（反映局部稠密程度）。

2) 最近邻距离变异系数 $CV=\sigma_r/\bar r$（反映是否有聚团/不均匀）。

3) Clark–Evans 最近邻指数 $R$：

- 观测最近邻均值 $r_{obs}=\bar r$；
- 在完全空间随机（CSR，Poisson）下，期望最近邻距离

$$
 r_{exp}=\frac{1}{2\sqrt{\lambda}},\quad \lambda=\frac{n}{|W|},\quad |W|=\pi R_c^2
$$

- 指数

$$
 R=\frac{r_{obs}}{r_{exp}}
$$

解释（近似）：
- $R\approx1$：随机（CSR）；
- $R<1$：团簇聚集；
- $R>1$：更均匀/规则。

对应代码：[src/analysis.py](src/analysis.py)

## 4 如何运行（Windows / PowerShell）

1) 安装依赖（建议使用你已有的 `.venv`）：

```powershell
cd e:\360MoveData\Users\admin\Desktop\holoholholi
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) 运行演示：

```powershell
python run_demo.py
```

3) 输出在 `outputs/`：
- `packing.glb`（三维堆积）
- `topdown.png`（俯视）
- `section.png`（截面）
- `section_metrics.csv`（几何指标表）
- `area_fit.png`、`deq_fit.png`（分布拟合图）
- `centroids.png`（质心散点）
- `summary.json`（汇总与拟合/空间指标）

## 5 可写入论文/答辩的“引用文本”（可直接引用）

下面是可直接放入论文/建模报告的“引用性描述”（你可按需改写）：

- **关于截面等效直径定义**：
  “将二维截面区域视为与其面积相等的圆，定义面积等效直径 $d_{eq}=\sqrt{4A/\pi}$，以消除截面形状差异对尺度描述的影响。”

- **关于圆度/紧致度指标**：
  “采用圆度 $C=4\pi A/P^2$ 衡量截面形状的紧致程度；当截面为完美圆形时 $C=1$，形状越不规则则 $C$ 越小。”

- **关于空间随机性检验（Clark–Evans）**：
  “将截面质心点视为二维点过程，在完全空间随机（CSR）假设下，最近邻距离期望为 $r_{exp}=1/(2\sqrt{\lambda})$。据此定义 Clark–Evans 指数 $R=r_{obs}/r_{exp}$，$R\approx 1$ 表示随机分布，$R<1$ 表示团簇聚集，$R>1$ 表示更均匀的排布。”

> 注：上述最近邻指数与解释来自经典空间点格局统计文献（见“参考文献”）。

## 6 代码结构说明

- [run_demo.py](run_demo.py)：一键生成堆积、截面、统计与图。
- [src/config.py](src/config.py)：参数配置（容器尺寸、粒径范围、沉降步长等）。
- [src/particle.py](src/particle.py)：不规则颗粒几何模型。
- [src/packing.py](src/packing.py)：圆柱内沉降式堆积仿真。
- [src/section.py](src/section.py)：平面截切并得到二维多边形。
- [src/analysis.py](src/analysis.py)：几何统计、分布拟合、空间分布指标。
- [src/plots.py](src/plots.py)：截面图、直方图+拟合曲线、质心散点图。

## 7 参考文献（建议在报告中列出）

1. Clark, P. J., & Evans, F. C. (1954). *Distance to Nearest Neighbor as a Measure of Spatial Relationships in Populations*. Ecology, 35(4), 445–453.
2. Massey, F. J. (1951). *The Kolmogorov–Smirnov Test for Goodness of Fit*. Journal of the American Statistical Association, 46(253), 68–78.
3. Limpert, E., Stahel, W. A., & Abbt, M. (2001). *Log-normal Distributions across the Sciences: Keys and Clues*. BioScience, 51(5), 341–352.
4. Weibull, W. (1951). *A Statistical Distribution Function of Wide Applicability*. Journal of Applied Mechanics, 18, 293–297.

---

## 说明（重要）

- 本实现是“竞赛建模版”的随机堆积仿真：强调可复现、可调参、能生成合理微结构统计；
- 若你需要更接近真实物理的接触摩擦/转动/压实过程，可进一步升级为 DEM（如 LIGGGHTS / YADE）并将真实 CT 的粒径分布作为输入。
