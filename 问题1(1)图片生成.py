import sys
sys.path.append('src')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import voronoi_generator
import particle_generator
import size_distribution
import particle_analysis
import trimesh
from scipy.spatial import ConvexHull
import math

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def generate_figure_1():
    print("Generating Figure 1...")
    fig = plt.figure(figsize=(16, 4))
    
    # (a) Poisson Seeds
    ax1 = fig.add_subplot(141, projection='3d')
    bounds = ((0, 200), (0, 200), (0, 200))
    # Density for 50 particles in 200^3 = 8e6 um3. 50/8e6 = 6.25e-6
    seeds = voronoi_generator.generate_poisson_seeds(bounds, 6.25e-6, random_seed=42)
    # Filter for visualization
    seeds_viz = np.array(seeds)
    ax1.scatter(seeds_viz[:,0], seeds_viz[:,1], seeds_viz[:,2], s=10, c='k')
    ax1.set_title("(a) Poisson Seeds")
    set_axes_equal(ax1)
    
    # (b) Voronoi Cell
    ax2 = fig.add_subplot(142, projection='3d')
    polyhedrons = voronoi_generator.generate_voronoi_polyhedrons(seeds)
    # Pick one in the center roughly
    if not polyhedrons:
        print("No polyhedrons generated!")
        return
        
    keys = list(polyhedrons.keys())
    idx = len(keys) // 2
    verts_b = np.array(polyhedrons[keys[idx]])
    
    # Center it
    centroid = np.mean(verts_b, axis=0)
    verts_b = verts_b - centroid
    
    hull_b = ConvexHull(verts_b)
    ax2.add_collection3d(Poly3DCollection(verts_b[hull_b.simplices], alpha=0.5, edgecolor='k'))
    ax2.set_title("(b) Voronoi Cell")
    set_axes_equal(ax2)
    
    # (c) Scaled
    ax3 = fig.add_subplot(143, projection='3d')
    # Scale to ~60um
    vol_b = hull_b.volume
    # Target D_eq approx 60. ln(60) ~ 4.09
    d_eq_target = size_distribution.generate_normal_particle_size(4.09, 0.01, (59, 61)) 
    s = d_eq_target / (6 * vol_b / np.pi)**(1/3)
    verts_c = verts_b * s
    hull_c = ConvexHull(verts_c)
    ax3.add_collection3d(Poly3DCollection(verts_c[hull_c.simplices], alpha=0.5, edgecolor='k'))
    ax3.set_title("(c) Scaled Skeleton")
    set_axes_equal(ax3)
    
    # (d) Perturbed
    ax4 = fig.add_subplot(144, projection='3d')
    verts_d = particle_generator.perlin_noise_modification(verts_c.tolist(), amplitude=2.5)
    verts_d = np.array(verts_d)
    hull_d = ConvexHull(verts_d)
    ax4.add_collection3d(Poly3DCollection(verts_d[hull_d.simplices], alpha=0.5, edgecolor='k', facecolor='cyan'))
    ax4.set_title("(d) Final Particle")
    set_axes_equal(ax4)
    
    plt.tight_layout()
    plt.savefig('outputs/figure_1(1)_modeling_process.png')
    plt.show()

def generate_figure_2():
    print("Generating Figure 2 (this may take time)...")
    amplitudes = np.linspace(1.0, 3.0, 5)
    psi_means, psi_stds = [], []
    r_means, r_stds = [], []
    
    # Base shape
    bounds = ((0, 100), (0, 100), (0, 100))
    seeds = voronoi_generator.generate_poisson_seeds(bounds, 5e-5, random_seed=42)
    polyhedrons = voronoi_generator.generate_voronoi_polyhedrons(seeds)
    base_keys = list(polyhedrons.keys())[:10] # Use 10 shapes
    
    for A in amplitudes:
        psis = []
        rs = []
        for _ in range(2): # 2 repetitions
            for k in base_keys:
                verts = np.array(polyhedrons[k])
                center = np.mean(verts, axis=0)
                verts = verts - center
                # Scale roughly
                verts = verts * (30 / (np.max(np.abs(verts)) + 1e-6))
                
                try:
                    verts_mod = particle_generator.perlin_noise_modification(verts.tolist(), amplitude=A)
                    params = particle_analysis.analyze_particle(verts_mod)
                    psis.append(params['Sphericity'])
                    rs.append(params['Roundness (Approx)'])
                except:
                    pass
        
        psi_means.append(np.mean(psis))
        psi_stds.append(np.std(psis))
        r_means.append(np.mean(rs))
        r_stds.append(np.std(rs))
        print(f"  A={A:.1f}: Psi={psi_means[-1]:.2f}, R={r_means[-1]:.2f}")
        
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Perlin Amplitude A (um)')
    ax1.set_ylabel('Sphericity', color=color)
    ax1.plot(amplitudes, psi_means, color=color, marker='o')
    ax1.fill_between(amplitudes, np.array(psi_means)-np.array(psi_stds), np.array(psi_means)+np.array(psi_stds), alpha=0.2, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Roundness', color=color)
    ax2.plot(amplitudes, r_means, color=color, linestyle='--', marker='s')
    ax2.fill_between(amplitudes, np.array(r_means)-np.array(r_stds), np.array(r_means)+np.array(r_stds), alpha=0.2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Feasible region
    # Psi < 0.75, R < 0.6
    # Shade region where condition met? Or just mark it.
    # The plot in paper has a shaded "Feasible Region".
    # We can just draw lines
    ax1.axhline(0.75, color='b', linestyle=':', label='Psi Limit')
    ax2.axhline(0.6, color='r', linestyle=':', label='R Limit')
    
    plt.title("Figure 2: Morphological Metrics vs A")
    plt.savefig('outputs/figure_1(1)_metrics_analysis.png')
    plt.show()

def generate_figure_3():
    print("Generating Figure 3...")
    # Find a particle with desired params or just generate one
    bounds = ((0, 100), (0, 100), (0, 100))
    seeds = voronoi_generator.generate_poisson_seeds(bounds, 5e-5, random_seed=123)
    poly = voronoi_generator.generate_voronoi_polyhedrons(seeds)
    keys = list(poly.keys())
    
    # Try a few to find a good looking one
    target_k = keys[len(keys)//2]
    verts = np.array(poly[target_k])
    verts = verts - np.mean(verts, axis=0)
    verts = verts * 20 # Scale up
    verts = particle_generator.perlin_noise_modification(verts.tolist(), amplitude=2.5)
    verts = np.array(verts)

    hull = ConvexHull(verts)
    mesh = trimesh.Trimesh(vertices=verts, faces=hull.simplices)
    
    fig = plt.figure(figsize=(15, 5))
    
    # (a) Geometry
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.add_collection3d(Poly3DCollection(verts[hull.simplices], alpha=1.0, linewidth=0.5, edgecolor='w'))
    set_axes_equal(ax1)
    ax1.set_title("(a) 3D Geometry")
    
    # (b) Inertia Axes
    ax2 = fig.add_subplot(132, projection='3d')
    # Recenter
    mesh.vertices -= mesh.center_mass
    # Inertia
    # Use mesh properties
    evals = mesh.principal_inertia_components
    evecs = mesh.principal_inertia_vectors
    
    # Vectors are normalized? Yes.
    # evals are moments of inertia. Smallest MOI -> Longest axis.
    
    idx_sorted = np.argsort(evals) # Smallest to largest
    # 0 -> Smallest I -> Long axis (Red)
    # 1 -> Mid I -> Mid axis (Green)
    # 2 -> Largest I -> Short axis (Blue)
    
    ax2.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces], alpha=0.3, edgecolor='k'))
    # Set axis length based on particle size (e.g. max extent)
    # mesh.extents gives bounding box size
    max_extent = np.max(mesh.extents)
    axis_len = max_extent * 0.7 # 70% of bounding box
    
    ax2.quiver(0,0,0, evecs[idx_sorted[0]][0], evecs[idx_sorted[0]][1], evecs[idx_sorted[0]][2], length=axis_len, color='r', label='L')
    ax2.quiver(0,0,0, evecs[idx_sorted[1]][0], evecs[idx_sorted[1]][1], evecs[idx_sorted[1]][2], length=axis_len*0.8, color='g', label='M')
    ax2.quiver(0,0,0, evecs[idx_sorted[2]][0], evecs[idx_sorted[2]][1], evecs[idx_sorted[2]][2], length=axis_len*0.6, color='b', label='I')
    set_axes_equal(ax2)
    ax2.set_title("(b) Inertia Axes")
    ax2.legend()
    
    # (c) Curvature
    ax3 = fig.add_subplot(133, projection='3d')
    # Calculate curvature color
    k = trimesh.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices, 1.0) # radius > 0 usually give better results than 0 (which is only neighbors)
    
    # Normalize with robust scaling (5th-95th percentile)
    k_abs = np.abs(k)
    v_min, v_max = np.percentile(k_abs, [5, 95])
    # Clip values
    c = np.clip(k_abs, v_min, v_max)
    # Normalize to 0-1
    c = (c - v_min) / (v_max - v_min + 1e-6)
    
    # We need to set face colors based on vertex colors?
    # Poly3DCollection takes face colors.
    # Average vertex colors to face
    face_colors = np.zeros(len(mesh.faces))
    for i, face in enumerate(mesh.faces):
        # face contains 3 vertex indices
        face_vals = [c[v_idx] for v_idx in face]
        face_colors[i] = np.mean(face_vals)
    
    from matplotlib import cm
    cmap = cm.jet # 'jet' or 'viridis' is often better for detail than 'hot' which can be very dark at bottom
    colors = cmap(face_colors)
    
    poly = Poly3DCollection(mesh.vertices[mesh.faces], facecolors=colors, edgecolor='none')
    ax3.add_collection3d(poly)
    set_axes_equal(ax3)
    ax3.set_title("(c) Curvature Heatmap")
    
    plt.tight_layout()
    plt.savefig('outputs/figure_1(1)_curvature_heatmap.png')
    plt.show()

def generate_figure_4():
    print("Generating Figure 4...")
    fig = plt.figure(figsize=(16, 4))
    
    # (a) Sphere
    ax1 = fig.add_subplot(141, projection='3d')
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = 30 * np.outer(np.cos(u), np.sin(v))
    y = 30 * np.outer(np.sin(u), np.sin(v))
    z = 30 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x, y, z, color='lightgrey', alpha=0.8)
    ax1.set_title("(a) Sphere")
    set_axes_equal(ax1)
    
    # (b) Super Ellipsoid
    ax2 = fig.add_subplot(142, projection='3d')
    # x = c * sgn(cos u) |cos u|^n * ...
    n = 0.5 # Cuboid-ish
    # Parametric equations
    # Using scale 30, 20, 15
    rx, ry, rz = 40, 30, 20
    
    def sgn(x): return np.sign(x)
    
    u = np.linspace(-np.pi, np.pi, 50)
    v = np.linspace(-np.pi/2, np.pi/2, 50)
    u, v = np.meshgrid(u, v)
    
    cu, su = np.cos(u), np.sin(u)
    cv, sv = np.cos(v), np.sin(v)
    
    sf1 = sgn(cv) * np.abs(cv)**n
    sf2 = sgn(sv) * np.abs(sv)**n
    
    x = rx * sf1 * (sgn(cu) * np.abs(cu)**n)
    y = ry * sf1 * (sgn(su) * np.abs(su)**n)
    z = rz * sf2
    
    ax2.plot_surface(x, y, z, color='lightblue', alpha=0.8)
    ax2.set_title("(b) Super-ellipsoid")
    set_axes_equal(ax2)
    
    # (c) Spherical Harmonics (Approx)
    ax3 = fig.add_subplot(143, projection='3d')
    # r = 30 + 5 * Y_lm...
    # Simple approx: r = 30 + 5 * sin(4u)cos(4v)
    r = 30 + 5 * np.sin(4*u) * np.cos(4*v)
    x = r * np.cos(u) * np.cos(v) # Wait, spherical coords mapping
    # Standard spherical:
    x = r * np.cos(u) * np.sin(v + np.pi/2) # Adapting to previous grid
    y = r * np.sin(u) * np.sin(v + np.pi/2)
    z = r * np.cos(v + np.pi/2)
    
    ax3.plot_surface(x, y, z, color='lightgreen', alpha=0.8)
    ax3.set_title("(c) Spherical Harmonics")
    set_axes_equal(ax3)
    
    # (d) Our Model
    ax4 = fig.add_subplot(144, projection='3d')
    # Generate one
    bounds = ((0, 100), (0, 100), (0, 100))
    seeds = voronoi_generator.generate_poisson_seeds(bounds, 5e-5, random_seed=999)
    poly = voronoi_generator.generate_voronoi_polyhedrons(seeds)
    verts = np.array(list(poly.values())[5])
    verts = verts - np.mean(verts, axis=0)
    verts = verts * 20
    verts = particle_generator.perlin_noise_modification(verts.tolist(), amplitude=2.5)
    verts = np.array(verts)
    
    hull = ConvexHull(verts)
    ax4.add_collection3d(Poly3DCollection(verts[hull.simplices], alpha=0.8, edgecolor='k', facecolor='cyan'))
    ax4.set_title("(d) Voronoi-Perlin")
    set_axes_equal(ax4)
    
    plt.tight_layout()
    plt.savefig('outputs/figure_1(1)_comparison.png')
    plt.show()

if __name__ == "__main__":
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
