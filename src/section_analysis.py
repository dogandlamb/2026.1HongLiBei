import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import lognorm, weibull_min, gamma, norm, kstest
import matplotlib.pyplot as plt

def get_slice_polygon(vertices, z_plane):
    """
    Compute the 2D intersection polygon of a convex polyhedron with a horizontal plane z = z_plane.
    Returns a list of [x, y] coordinates for the polygon vertices, ordered counter-clockwise.
    Returns None if no intersection.
    """
    verts = np.array(vertices)
    z_min = np.min(verts[:, 2])
    z_max = np.max(verts[:, 2])
    
    # Check bounds
    if z_plane <= z_min or z_plane >= z_max:
        return None
        
    try:
        hull = ConvexHull(verts)
    except:
        return None

    # Find edges that cross the plane
    # Hull simplices are triangles (faces).
    # We can iterate over all edges of the hull.
    # A robust way is to iterate over simplices, find edges crossing z_plane.
    
    crossing_points = []
    
    # Use a set of sorted edge tuples to avoid duplicate processing
    processed_edges = set()
    
    for simplex in hull.simplices:
        # Simplex is indices of 3 vertices
        for i in range(3):
            idx1, idx2 = simplex[i], simplex[(i+1)%3]
            
            # Ensure unique edge check
            edge_key = tuple(sorted((idx1, idx2)))
            if edge_key in processed_edges:
                continue
            processed_edges.add(edge_key)
            
            p1 = verts[idx1]
            p2 = verts[idx2]
            
            z1, z2 = p1[2], p2[2]
            
            # Check for crossing
            if (z1 < z_plane and z2 > z_plane) or (z1 > z_plane and z2 < z_plane):
                # Linear interpolation
                t = (z_plane - z1) / (z2 - z1)
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                crossing_points.append([x, y])
            
            # Handle exact match (rare with floats, but good to know)
            # If z1 == z_plane, strictly it touches. We can ignore or include.
            # For robust slicing of "solids", usually strict crossing is enough for the "inside".
            
    if len(crossing_points) < 3:
        return None
        
    points = np.array(crossing_points)
    
    # Sort points to form a polygon
    # Calculate centroid
    centroid = np.mean(points, axis=0)
    # Atan2 for angle
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    # Sort
    sort_order = np.argsort(angles)
    polygon = points[sort_order]
    
    return polygon

def calculate_geometry_properties(polygon):
    """
    Calculate Area, Perimeter, Equivalent Diameter, Circularity.
    Polygon is Nx2 numpy array.
    """
    if polygon is None or len(polygon) < 3:
        return None
        
    # Shoelace Formula for Area
    x = polygon[:, 0]
    y = polygon[:, 1]
    # Shift arrays for i+1
    x_shift = np.roll(x, -1)
    y_shift = np.roll(y, -1)
    
    area = 0.5 * np.abs(np.dot(x, y_shift) - np.dot(y, x_shift))
    
    # Perimeter
    dx = x - x_shift
    dy = y - y_shift
    perimeter = np.sum(np.sqrt(dx**2 + dy**2))
    
    # Equivalent Diameter
    d_eq = np.sqrt(4 * area / np.pi)
    
    # Circularity
    if perimeter == 0:
        c = 0
    else:
        c = (4 * np.pi * area) / (perimeter**2)
        
    return {
        'Area': area,
        'Perimeter': perimeter,
        'Equivalent Diameter': d_eq,
        'Circularity': c
    }

def perform_statistical_analysis(data_list):
    """
    Fit LogNormal and Weibull distributions to the data.
    Perform KS test.
    Returns dictionary of results.
    """
    data = np.array(data_list)
    # Remove NaN or non-positive values if log-fitting
    data = data[data > 0] 
    
    if len(data) < 2:
        return None

    results = {}
    
    # 1. Log-normal Fit
    try:
        shape_ln, loc_ln, scale_ln = lognorm.fit(data, floc=0) 
        results['LogNormal'] = {
            'mu': np.log(scale_ln),
            'sigma': shape_ln,
            'ks_stat': kstest(data, 'lognorm', args=(shape_ln, loc_ln, scale_ln))[0],
            'p_value': kstest(data, 'lognorm', args=(shape_ln, loc_ln, scale_ln))[1],
            'params_scipy': (shape_ln, loc_ln, scale_ln)
        }
    except Exception as e:
        results['LogNormal'] = {'p_value': 0, 'error': str(e)}

    # 2. Weibull Fit
    try:
        shape_wb, loc_wb, scale_wb = weibull_min.fit(data, floc=0)
        results['Weibull'] = {
            'k (shape)': shape_wb,
            'lambda (scale)': scale_wb,
            'ks_stat': kstest(data, 'weibull_min', args=(shape_wb, loc_wb, scale_wb))[0],
            'p_value': kstest(data, 'weibull_min', args=(shape_wb, loc_wb, scale_wb))[1],
            'params_scipy': (shape_wb, loc_wb, scale_wb)
        }
    except Exception as e:
        results['Weibull'] = {'p_value': 0, 'error': str(e)}

    # 3. Gamma Fit
    try:
        a_gm, loc_gm, scale_gm = gamma.fit(data, floc=0)
        results['Gamma'] = {
            'alpha': a_gm,
            'scale': scale_gm,
            'ks_stat': kstest(data, 'gamma', args=(a_gm, loc_gm, scale_gm))[0],
            'p_value': kstest(data, 'gamma', args=(a_gm, loc_gm, scale_gm))[1],
            'params_scipy': (a_gm, loc_gm, scale_gm)
        }
    except Exception as e:
        results['Gamma'] = {'p_value': 0, 'error': str(e)}

    # 4. Normal Fit
    try:
        mu_norm, std_norm = norm.fit(data)
        results['Normal'] = {
            'mu': mu_norm,
            'std': std_norm,
            'ks_stat': kstest(data, 'norm', args=(mu_norm, std_norm))[0],
            'p_value': kstest(data, 'norm', args=(mu_norm, std_norm))[1],
            'params_scipy': (mu_norm, std_norm)
        }
    except Exception as e:
        results['Normal'] = {'p_value': 0, 'error': str(e)}
    
    return results

def plot_distributions(data_list, results, title, xlabel, save_path=None):
    if results is None: return

    data = np.array(data_list)
    data = data[data > 0]
    
    plt.figure(figsize=(10, 6))
    
    # Histogram
    count, bins, ignored = plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Histogram')
    
    if len(bins) < 2: return
    x = np.linspace(min(bins), max(bins), 200)
    
    # Plot LogNormal
    if 'params_scipy' in results['LogNormal']:
        s, loc, scale = results['LogNormal']['params_scipy']
        current_pdf = lognorm.pdf(x, s, loc, scale)
        if not np.any(np.isnan(current_pdf)):
             plt.plot(x, current_pdf, 'r-', linewidth=2, label=f'LogNormal (p={results["LogNormal"]["p_value"]:.3f})')
    
    # Plot Weibull
    if 'params_scipy' in results['Weibull']:
        c, loc, scale = results['Weibull']['params_scipy']
        current_pdf = weibull_min.pdf(x, c, loc, scale)
        if not np.any(np.isnan(current_pdf)):
             plt.plot(x, current_pdf, 'b--', linewidth=2, label=f'Weibull (p={results["Weibull"]["p_value"]:.3f})')
    
    # Plot Gamma
    if 'params_scipy' in results.get('Gamma', {}):
        a, loc, scale = results['Gamma']['params_scipy']
        current_pdf = gamma.pdf(x, a, loc, scale)
        if not np.any(np.isnan(current_pdf)):
             plt.plot(x, current_pdf, 'y-.', linewidth=2, label=f'Gamma (p={results["Gamma"]["p_value"]:.3f})')

    # Plot Normal
    if 'params_scipy' in results.get('Normal', {}):
        loc, scale = results['Normal']['params_scipy']
        current_pdf = norm.pdf(x, loc, scale)
        if not np.any(np.isnan(current_pdf)):
             plt.plot(x, current_pdf, 'm:', linewidth=2, label=f'Normal (p={results["Normal"]["p_value"]:.3f})')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    # Note: plt.show() should be called outside if needed.
