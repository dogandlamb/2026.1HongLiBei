import numpy as np

def support_polyhedron(pts, direction):
    """
    Returns the vertex of the polyhedron 'pts' that is furthest in 'direction'.
    pts: (N, 3) numpy array
    direction: (3,) numpy array
    """
    dot_products = np.dot(pts, direction)
    idx = np.argmax(dot_products)
    return pts[idx]

def get_support(pts1, pts2, direction):
    """
    Returns the support point of the Minkowski difference (M = A - B)
    in the given direction.
    """
    return support_polyhedron(pts1, direction) - support_polyhedron(pts2, -direction)

def gjk(pts1, pts2, max_iterations=20):
    """
    GJK algorithm to detect collision between two convex polyhedrons.
    Returns (is_colliding, simplex)
    """
    # Initial direction (arbitrary, e.g., difference of centroids)
    center1 = np.mean(pts1, axis=0)
    center2 = np.mean(pts2, axis=0)
    d = center2 - center1
    if np.all(d == 0):
        d = np.array([1.0, 0, 0])

    a = get_support(pts1, pts2, d)
    simplex = [a]
    d = -a

    for _ in range(max_iterations):
        a = get_support(pts1, pts2, d)
        
        # If the furthest point in the direction 'd' hasn't passed the origin,
        # then the origin cannot be inside the Minkowski sum.
        if np.dot(a, d) < 0:
            return False, []

        simplex.append(a)
        
        if do_simplex(simplex, d):
            return True, simplex
            
    return False, simplex # Usually shouldn't reach here if loop is sufficient

def do_simplex(simplex, d):
    """
    Updates the simplex and search direction 'd'.
    Returns True if origin is enclosed (collision found).
    """
    # This implementation handles Line (2 pts), Triangle (3 pts), Tetrahedron (4 pts)
    # Simplex points are added such that the last added point is at index -1 (A)
    
    a = simplex[-1]
    ao = -a # Vector from A to Origin
    
    if len(simplex) == 2: # Line segment
        b = simplex[0]
        ab = b - a
        d[:] = np.cross(np.cross(ab, ao), ab)
        if np.all(d == 0): 
             d[:] = np.cross(ab, np.array([1, 0, 0])) 
             if np.all(d == 0): d[:] = np.cross(ab, np.array([0, 1, 0]))
        return False

    elif len(simplex) == 3: # Triangle
        b = simplex[1]
        c = simplex[0]
        ab = b - a
        ac = c - a
        abc = np.cross(ab, ac) 
        
        if np.dot(np.cross(abc, ac), ao) > 0:
            if np.dot(ac, ao) > 0:
                simplex[:] = [c, a]
                d[:] = np.cross(np.cross(ac, ao), ac)
            else:
                if np.dot(np.cross(ab, abc), ao) > 0:
                     simplex[:] = [b, a]
                     d[:] = np.cross(np.cross(ab, ao), ab)
                else:
                    simplex[:] = [c, b, a]
                    if np.dot(abc, ao) > 0: d[:] = abc
                    else:
                        simplex[:] = [b, c, a] 
                        d[:] = -abc
        else:
             if np.dot(np.cross(ab, abc), ao) > 0:
                 if np.dot(ab, ao) > 0:
                     simplex[:] = [b, a]
                     d[:] = np.cross(np.cross(ab, ao), ab)
                 else:
                     simplex[:] = [c, a]
                     d[:] = np.cross(np.cross(ac, ao), ac)
             else:
                 if np.dot(abc, ao) > 0: d[:] = abc
                 else:
                     simplex[:] = [b, c, a]
                     d[:] = -abc
        return False
        
    elif len(simplex) == 4: # Tetrahedron
        b = simplex[2]
        c = simplex[1]
        d_pt = simplex[0] 
        ab = b - a
        ac = c - a
        ad = d_pt - a
        abc = np.cross(ab, ac)
        acd = np.cross(ac, ad)
        adb = np.cross(ad, ab)
        
        if np.dot(abc, ao) > 0:
            simplex[:] = [c, b, a]
            d[:] = abc
            return False
            
        if np.dot(acd, ao) > 0:
            simplex[:] = [d_pt, c, a]
            d[:] = acd
            return False
            
        if np.dot(adb, ao) > 0:
            simplex[:] = [b, d_pt, a]
            d[:] = adb
            return False
            
        return True

    return False

def epa(pts1, pts2, simplex):
    """
    Expanding Polytope Algorithm to find penetration depth and normal.
    Returns (depth, normal_vector)
    """
    faces = []
    # Initial simplex winding check: needs to be CCW relative to outside
    # We rely on do_simplex output which might not be perfect tetrahedron winding
    # Just construct and fix normals
    v0, v1, v2, v3 = simplex[0], simplex[1], simplex[2], simplex[3]
    polytope = [[v0, v1, v2], [v0, v2, v3], [v0, v3, v1], [v1, v3, v2]]
    
    tol = 1e-4
    max_iters = 30
    
    # Helper to get dist and normal of face
    def get_info(face):
        n = np.cross(face[1]-face[0], face[2]-face[0])
        l = np.linalg.norm(n)
        if l == 0: return np.inf, n
        n = n/l
        d = np.dot(n, face[0])
        # Origin is inside, so d should be positive if Normal points OUT?
        # If Normal points OUT, and Origin is (0,0,0), then plane eq is defined by P dot N = d.
        # Vector from origin to plane is d * N.
        # But if origin is inside, d must be < 0 if N points towards origin.
        # Wait, usually plane dist is P.N. If P.N > 0 for points on plane, and Origin is 0
        # If origin inside, P.N > 0 (assuming N points away from origin).
        if d < 0: n, d = -n, -d
        return d, n

    for _ in range(max_iters):
        min_dist = np.inf
        best_n = None
        closest_idx = -1
        
        normals = []
        dists = []
        
        for i, f in enumerate(polytope):
            d, n = get_info(f)
            dists.append(d)
            normals.append(n)
            if d < min_dist:
                min_dist, best_n, closest_idx = d, n, i
                
        if min_dist == np.inf: break
        
        support = get_support(pts1, pts2, best_n)
        if abs(np.dot(support, best_n) - min_dist) < tol:
            return min_dist, best_n
            
        # Refine
        edges = []
        new_poly = []
        for i, f in enumerate(polytope):
            if np.dot(normals[i], support) > dists[i] + tol: # Visible
                for a, b in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
                    edge = sorted((tuple(a), tuple(b))) # Sort to identify unique
                    edge = tuple(edge)
                    if edge in edges: edges.remove(edge)
                    else: edges.append(edge)
            else:
                new_poly.append(f)
                
        for e in edges:
            new_poly.append([np.array(e[0]), np.array(e[1]), support])
        polytope = new_poly
        
    return min_dist, best_n
