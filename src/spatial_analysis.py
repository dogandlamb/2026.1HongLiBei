import numpy as np
from scipy.spatial.distance import pdist, squareform

def calculate_ripleys_k_circle(points, r_values, domain_radius):
    """
    Calculate Ripley's K function for a set of points inside a circular domain
    with isotropic edge correction.
    
    Args:
        points: (N, 2) array of point coordinates.
        r_values: array of distances to evaluate K(r) at.
        domain_radius: radius of the circular domain (R).
        
    Returns:
        k_values: array of K(r) values.
    """
    points = np.asarray(points)
    n = len(points)
    area = np.pi * domain_radius**2
    lambda_density = n / area
    
    if n < 2:
        return np.zeros_like(r_values)

    # Distances between all pairs
    dists = squareform(pdist(points))
    
    # Distances from center of domain
    dists_from_center = np.linalg.norm(points, axis=1)
    
    k_values = []
    
    for r in r_values:
        # Indicator matrix for d_ij <= r (ignoring diagonal)
        # Using floating point comparison tolerance could be wise, but standard inequality is usually fine
        indicator = (dists <= r) & (dists > 0)
        
        # Calculate weights for edge correction
        # For each point i, and neighbor j
        # Weight w_ij depends on distance of i from center (b_i) and distance d_ij
        
        # We need to iterate to apply weights correctly per pair
        # Vectorized implementation:
        
        # For each pair (i, j), we have dists[i, j] and dists_from_center[i]
        # Let d = dists[i, j]
        # Let b = dists_from_center[i]
        
        # Broadcast b to shape (N, N) where rows are i
        b_matrix = dists_from_center[:, np.newaxis] # Shape (N, 1)
        d_matrix = dists # Shape (N, N)
        
        # Check if circle of radius d centered at i is fully inside the domain
        # Condition: b + d <= domain_radius
        fully_inside = (b_matrix + d_matrix) <= domain_radius
        
        weights = np.ones((n, n))
        
        # Calculate weights for those not fully inside
        # Only calculate where d_ij <= r, otherwise weight doesn't matter (indicator is 0)
        mask_correction = (~fully_inside) & indicator
        
        if np.any(mask_correction):
            # We need to broadcast b_matrix to (N, N) to index it with mask_correction
            b_full = np.broadcast_to(b_matrix, (n, n))
            
            # Formula for angle gamma:
            # cos(gamma) = (b^2 + d^2 - R^2) / (2bd)
            b = b_full[mask_correction]
            d = d_matrix[mask_correction]

            R = domain_radius
            
            cos_gamma = (b**2 + d**2 - R**2) / (2 * b * d)
            
            # Clip for numerical stability
            cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
            
            gamma = np.arccos(cos_gamma)
            
            # The angle inside the domain is 2*gamma ? 
            # Wait, verify geometry.
            # Law of cosines gives angle of the triangle (Origin, Point i, Intersection).
            # The angle at Point i is gamma.
            # The intersection points are symmetric, so total angle subtended by the intersection chord is 2*gamma?
            # NO. 
            # If b+d > R, the circle centered at i extends outside.
            # The angle gamma we calculated is between the vector to the origin and the intersection point.
            # Since the domain is convex and contains the center of the measuring circle (point i),
            # the arc *inside* the domain is the major arc if the center is close to boundary? 
            # No, simple intersection of two circles.
            # The angle of the arc INSIDE the domain (centered at i) is 2 * gamma.
            # The weight is 2*pi / (angle_inside).
            # So w = 2*pi / (2*gamma) = pi / gamma.
            
            w_partial = np.pi / gamma
            weights[mask_correction] = w_partial
            
        # Sum weighted counts
        # sum_{i} sum_{j!=i} w_ij * I(d_ij <= r)
        weighted_count = np.sum(weights * indicator)
        
        k_val = (area / n**2) * weighted_count
        k_values.append(k_val)
        
    return np.array(k_values)

def calculate_l_function(k_values, r_values):
    """
    Calculate Besag's L function.
    L(r) = sqrt(K(r)/pi) - r
    """
    return np.sqrt(k_values / np.pi) - r_values

def generate_csr_points(n, radius):
    """
    Generate n points uniformly distributed in a circle of given radius.
    """
    # Generate in polar coords to ensure uniformity
    r = radius * np.sqrt(np.random.random(n))
    theta = np.random.random(n) * 2 * np.pi
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return np.column_stack((x, y))

def monte_carlo_envelope(n_points, r_values, domain_radius, n_simulations=50):
    """
    Perform Monte Carlo simulations to generate confidence envelope for CSR.
    Returns:
        l_min: minimum L(r) across simulations
        l_max: maximum L(r) across simulations
        l_mean: mean L(r)
    """
    all_l_values = []
    
    for _ in range(n_simulations):
        # Generate random points
        random_points = generate_csr_points(n_points, domain_radius)
        
        # Calculate K and L
        k_sim = calculate_ripleys_k_circle(random_points, r_values, domain_radius)
        l_sim = calculate_l_function(k_sim, r_values)
        
        all_l_values.append(l_sim)
    
    all_l_values = np.array(all_l_values)
    
    l_min = np.min(all_l_values, axis=0)
    l_max = np.max(all_l_values, axis=0)
    l_mean = np.mean(all_l_values, axis=0)
    
    return l_min, l_max, l_mean
