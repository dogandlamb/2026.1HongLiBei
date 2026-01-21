import numpy as np
from scipy.spatial import ConvexHull

def calculate_pca_dimensions(vertices):
    """
    Use PCA to estimate Long, Intermediate, and Short axes of the particle.
    Returns (S, I, L).
    """
    points = np.array(vertices)
    if len(points) < 4:
        # Not enough points for 3D convex hull or meaningful PCA usually
        return 0, 0, 0

    # Centroid
    centroid = np.mean(points, axis=0)
    # Center points
    centered_points = points - centroid
    # Covariance Matrix
    cov_matrix = np.cov(centered_points, rowvar=False)
    # Eigenvalues and Eigenvectors
    # eigh returns eigenvalues in ascending order
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    except:
        return 0, 0, 0
    
    # Project points onto eigenvectors
    rotated_data = np.dot(centered_points, eigenvectors)
    
    # Ranges along axes
    min_vals = np.min(rotated_data, axis=0)
    max_vals = np.max(rotated_data, axis=0)
    lengths = max_vals - min_vals
    
    # eigenvalues sorted ascending -> lengths[0] is Short, lengths[1] is Intermediate, lengths[2] is Long
    S = lengths[0]
    I = lengths[1]
    L = lengths[2]
    
    return S, I, L

import trimesh

def analyze_particle(vertices):
    """
    Calculates geometric parameters for a set of 3D vertices (polyhedron).
    """
    try:
        hull = ConvexHull(vertices)
        volume = hull.volume
        area = hull.area
        
        # Enhanced Roundness Calculation using Trimesh
        # Construct trimesh object
        # Note: Hull simplices point outwards, good for mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=hull.simplices)
        
        # Calculate vertex areas (barycentric)
        vertex_areas = np.zeros(len(mesh.vertices))
        for i, face in enumerate(mesh.faces):
            f_area = mesh.area_faces[i]
            # Distribute area to 3 vertices
            for v_idx in face:
                vertex_areas[v_idx] += f_area / 3.0
                
        # Gaussian curvature (angle deficit)
        k_integrated = mesh.vertex_defects
        
        # Curvature density K = k_integrated / area
        # Protect against small areas
        vertex_areas[vertex_areas < 1e-12] = 1e-12
        K = k_integrated / vertex_areas
        
        # Local radius of curvature r_i = 1 / sqrt(|K|)
        # The paper says r_i = 1 / |K_G|. If K_G is density (1/L^2), then r_i is L^2?
        # If K_G is 1/L like mean curvature, then r_i is L.
        # Assuming r_i is a length, and K is 1/L^2. Then r_i = 1/sqrt(K).
        # If the paper says r_i = 1/K_G, maybe K_G is defined as 1/R?
        # Let's use 1 / sqrt(|K|) to be dimensionally consistent (Length).
        
        K_abs = np.abs(K)
        # Handle flat regions (K ~ 0) -> Large radius
        # Cap radius at reasonable max (e.g. 5 * D_eq) to prevent outliers skewing mean
        K_abs[K_abs < 1e-6] = 1e-6 
        r_i = 1.0 / np.sqrt(K_abs)
        
        # Roundness definition from paper
        R_max = np.max(r_i)
        if R_max > 0:
            roundness = np.mean(r_i) / R_max
        else:
            roundness = 0
            
    except Exception as e:
        # Fallback
        volume = 0
        area = 0
        roundness = 0
        print(f"Analysis failed: {e}")

        return {
            "Equivalent Diameter": 0,
            "Sphericity": 0,
            "Flatness": 0,
            "Elongation": 0,
            "Roundness": 0
        }
    
    # Equivalent Diameter
    eq_diameter = (6 * volume / np.pi) ** (1/3)
    
    # Sphericity
    # Psi = (pi^(1/3) * (6V)^(2/3)) / A
    numerator = (np.pi ** (1/3)) * ((6 * volume) ** (2/3))
    sphericity = numerator / area if area > 0 else 0
    
    # Dimensions
    S, I, L = calculate_pca_dimensions(vertices)
    
    # Flatness (S/I)
    flatness = S / I if I > 0 else 0
    
    # Elongation (L/I)
    elongation = L / I if I > 0 else 0
    
    # Roundness
    # Calculated above
    
    return {
        "Equivalent Diameter": eq_diameter,
        "Sphericity": sphericity,
        "Flatness": flatness,
        "Elongation": elongation,
        "Roundness (Approx)": roundness
    }
