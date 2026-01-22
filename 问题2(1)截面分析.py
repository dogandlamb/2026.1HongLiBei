import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# 添加 src 到路径
sys.path.append('src')
import section_analysis

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def main():
    print("======== 截面几何特征分析 (问题2) ========")
    
    # 1. 加载数据
    data_path = 'outputs/particles_data.pkl'
    if not os.path.exists(data_path):
        print(f"错误: 未找到数据文件 {data_path}")
        print("请先运行 '问题1(2)堆积模拟.py' 以生成模拟数据。")
        return

    print(f"正在从 {data_path} 加载颗粒数据...")
    with open(data_path, 'rb') as f:
        particles_vertices = pickle.load(f)
        
    print(f"已加载 {len(particles_vertices)} 个颗粒。")
    
    # 2. 确定截面高度 (z_0)
    # 获取堆积体的最大高度
    max_zs = [np.max(p[:, 2]) for p in particles_vertices]
    if not max_zs:
        print("错误: 数据中没有颗粒。")
        return
        
    pile_top = np.percentile(max_zs, 95)
    
    # 截面高度：取堆积体中部
    z_cut = pile_top * 0.5
    print(f"堆积体估算高度: {pile_top:.2f} um")
    print(f"选定的裁切截面高度 z_0: {z_cut:.2f} um")
    
    # 3. 执行切片和几何计算
    print("正在处理切片中...")
    slice_polygons = []
    
    # 用于统计的列表
    d_eq_list = []
    area_list = []
    perimeter_list = []
    circularity_list = []

    for i, verts in enumerate(particles_vertices):
        polygon = section_analysis.get_slice_polygon(verts, z_cut)
        
        if polygon is not None:
            slice_polygons.append(polygon)
            props = section_analysis.calculate_geometry_properties(polygon)
            if props:
                d_eq_list.append(props['Equivalent Diameter'])
                area_list.append(props['Area'])
                perimeter_list.append(props['Perimeter'])
                circularity_list.append(props['Circularity'])

    print(f"在该截面处共生成了 {len(slice_polygons)} 个有效的颗粒截斑。")
    
    if len(slice_polygons) < 3:
        print("警告: 找到的截斑太少，无法进行可靠的统计分析。")
        if len(slice_polygons) == 0: return

    # 4. 统计分析与绘图 (针对每个指标)
    metrics = [
        ('等效直径 (Equivalent Diameter)', d_eq_list, 'um', 'figure_2_size_dist.png'),
        ('截面面积 (Area)', area_list, 'um^2', 'figure_2_area_dist.png'),
        ('周长 (Perimeter)', perimeter_list, 'um', 'figure_2_perimeter_dist.png'),
        ('圆度 (Circularity)', circularity_list, '无量纲', 'figure_2_circularity_dist.png')
    ]
    
    for name, data, unit, filename in metrics:
        print(f"\n--- 分析指标: {name} ---")
        results = section_analysis.perform_statistical_analysis(data)
        if results is None:
            print(f"数据不足，无法分析 {name}")
            continue
            
        print(f"平均值: {np.mean(data):.4f} {unit}")
        if 'p_value' in results['LogNormal']:
            p_val = results['LogNormal']['p_value']
            print(f"对数正态分布 (LogNormal) P值: {p_val:.4e} " + ("-> 拟合良好" if p_val > 0.05 else "-> 拟合欠佳"))
        if 'p_value' in results['Weibull']:
            p_val = results['Weibull']['p_value']
            print(f"韦伯分布 (Weibull) P值:     {p_val:.4e} " + ("-> 拟合良好" if p_val > 0.05 else "-> 拟合欠佳"))
        if 'p_value' in results.get('Gamma', {}):
             p_val = results['Gamma']['p_value']
             print(f"伽马分布 (Gamma) P值:       {p_val:.4e} " + ("-> 拟合良好" if p_val > 0.05 else "-> 拟合欠佳"))
        if 'p_value' in results.get('Normal', {}):
             p_val = results['Normal']['p_value']
             print(f"正态分布 (Normal) P值:      {p_val:.4e} " + ("-> 拟合良好" if p_val > 0.05 else "-> 拟合欠佳"))
            
        save_path = os.path.join('outputs', filename)
        # 调用更新后的 plot_distributions，传入 title 和 xlabel
        section_analysis.plot_distributions(
            data, results, 
            title=f'{name} 分布拟合', 
            xlabel=f'{name} ({unit})', 
            save_path=save_path
        )

    # 5. 截面可视化
    plt.figure(figsize=(8, 8))
    for poly in slice_polygons:
        # 闭合多边形首尾相连
        poly_closed = np.vstack([poly, poly[0]])
        plt.plot(poly_closed[:, 0], poly_closed[:, 1], 'k-', linewidth=1)
        plt.fill(poly_closed[:, 0], poly_closed[:, 1], 'gray', alpha=0.5)
        
    plt.title(f'z={z_cut:.1f} um 处的二维截面图')
    plt.xlabel('X (um)')
    plt.ylabel('Y (um)')
    plt.axis('equal')
    
    # 绘制容器轮廓作为参考
    circle = plt.Circle((0, 0), 500, color='b', fill=False, linestyle='--', label='容器壁')
    plt.gca().add_patch(circle)
    plt.legend()
    
    plt.savefig('outputs/figure_2_section_view.png')
    
    print("\n分析完成。所有生成的图片已保存至 outputs/ 文件夹。")
    print("现在显示图表窗口...")
    plt.show()

if __name__ == "__main__":
    main()
