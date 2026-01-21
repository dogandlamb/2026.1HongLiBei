import numpy as np

def support_polyhedron(vertices, direction):
    """
    Find the furthest point in a specific direction for a convex polyhedron.
    :param vertices: (N, 3) numpy array of points
    :param direction: (3,) numpy array, direction vector
    :return: (3,) numpy array, the support point
    """
    # Using dot product to project points onto the direction vector
    dots = np.dot(vertices, direction)
    idx = np.argmax(dots)
    return vertices[idx]

def get_support(shape1_verts, shape2_verts, direction):
    """
    Compute the support point of the Minkowski difference (A - B)
    """
    s1 = support_polyhedron(shape1_verts, direction)
    s2 = support_polyhedron(shape2_verts, -direction)
    return s1 - s2

def gjk(shape1_verts, shape2_verts, max_iter=50):
    """
    GJK Algorithm to detect collision between two convex polyhedra.
    
    :param shape1_verts: Vertices of shape 1
    :param shape2_verts: Vertices of shape 2
    :return: (is_colliding, simplex)
    """
    # Initial direction (arbitrary, usually center to center or x-axis)
    # Using a fixed direction or random can be fine, let's pick X initially
    direction = np.array([1.0, 0.0, 0.0])
    
    # First point
    c = get_support(shape1_verts, shape2_verts, direction)
    simplex = [c]
    
    # Next direction is towards the origin
    direction = -c
    
    for _ in range(max_iter):
        if np.linalg.norm(direction) < 1e-6:
             # Origin is within tolerance
             return True, simplex
             
        a = get_support(shape1_verts, shape2_verts, direction)
        
        # If the new point is not past the origin along the direction, 
        # then the origin is not inside the Minkowski Sum.
        if np.dot(a, direction) < 0:
            return False, simplex  # No collision
        
        simplex.append(a)
        
        # Update simplex and direction
        if handle_simplex(simplex, direction):
            return True, simplex
            
    return False, simplex # Iteration limit reached, likely no collision or edge case

def handle_simplex(simplex, direction):
    """
    Updates the simplex and direction. 
    Returns True if the origin is enclosed (collision).
    Modifies 'simplex' and 'direction' in place (or via object reference).
    """
    # We are only dealing with Line, Triangle, Tetrahedron cases here
    # Point case is handled in initialization
    
    # Last point added
    a = simplex[-1]
    ao = -a # Direction towards origin
    
    if len(simplex) == 2: # Line segment
        b = simplex[0]
        ab = b - a
        
        # Determine if origin is closest to AB or A 
        if np.dot(ab, ao) > 0:
            # Origin is in the region of AB
            # New direction is perpendicular to AB towards Origin
            # vector_triple_product(ab, ao, ab) gives vector perpendicular to AB pointing to AO
            # But simpler: cross product
            # direction[:] = np.cross(np.cross(ab, ao), ab) # This might be zero if colinear
            
            # More stable approach for line:
            # Project AO onto AB? No, we need a direction. 
            # Perpendicular to AB closest to Origin.
            # Using triple product: (AB x AO) x AB
            direction[:] = np.cross(np.cross(ab, ao), ab)
        else:
            # Origin is closest to A (region beyond A)
            simplex.clear()
            simplex.append(a)
            direction[:] = ao
            
    elif len(simplex) == 3: # Triangle
        b = simplex[1]
        c = simplex[0]
        
        ab = b - a
        ac = c - a
        
        # Normals
        abc_norm = np.cross(ab, ac) # Normal to triangle
        
        # Edge normals (pointing outwards from triangle, on the plane)
        # We need to check regions: AB-out, AC-out, or Above/Below triangle
        
        # Check AB-out: (AB x n) . AO > 0 ? NO.
        # Standard way:
        # Normal to AB in plane of triangle, pointing away from C: np.cross(ab, abc_norm)
        
        if np.dot(np.cross(abc_norm, ac), ao) > 0:
            # Outside AC
            if np.dot(ac, ao) > 0:
                # In AC region
                simplex.clear()
                simplex.append(c)
                simplex.append(a)
                direction[:] = np.cross(np.cross(ac, ao), ac)
            else:
                 # In A star-region? Check AB
                 # Actually simpler to fall through or check star region of A
                 # Let's use the standard "reduction" logic
                 if np.dot(ab, ao) > 0:
                     simplex.clear()
                     simplex.append(b)
                     simplex.append(a)
                     direction[:] = np.cross(np.cross(ab, ao), ab)
                 else:
                     simplex.clear()
                     simplex.append(a)
                     direction[:] = ao
        elif np.dot(np.cross(ab, abc_norm), ao) > 0:
            # Outside AB
            if np.dot(ab, ao) > 0:
                simplex.clear()
                simplex.append(b)
                simplex.append(a)
                direction[:] = np.cross(np.cross(ab, ao), ab)
            else:
                simplex.clear()
                simplex.append(a)
                direction[:] = ao
        else:
            # Above or below triangle
            if np.dot(abc_norm, ao) > 0:
                direction[:] = abc_norm
            else:
                # Ensure CCW/Winding? Not strictly necessary for Tetrahedron GJK if we fix it later
                # But for EPA it's good to keep consistent winding. 
                # Let's just flip direction
                direction[:] = -abc_norm
                
                # Swap b and c to maintain winding if needed for your expansion
                simplex[0], simplex[1] = simplex[1], simplex[0] 

    elif len(simplex) == 4: # Tetrahedron
        b = simplex[2]
        c = simplex[1]
        d = simplex[0]
        
        # Faces: ABC, ACD, ADB (A is tip)
        # We know origin is "inside" relative to the base BCD because we came from there? NO.
        # We just added A. We need to check the 3 new faces connecting to A.
        
        ab = b - a
        ac = c - a
        ad = d - a
        
        # Face normals:
        abc_n = np.cross(ab, ac)
        acd_n = np.cross(ac, ad)
        adb_n = np.cross(ad, ab)
        
        # Check if origin is outside any face
        if np.dot(abc_n, ao) > 0:
            # Outside ABC
            # Remove D
            simplex.remove(d) 
            direction[:] = abc_n
            # Recurse/Check if we need to reduce to edge? 
            # In 3D GJK, if we verify edges when building triangle, we generally only need to check faces here.
            # However, robust implementations might re-check edges.
            # For compactness, assume triangle case handles edges next Iteration?
            # NO, handle_simplex must return valid next simplex/direction.
            # If we reduce to triangle, next loop calls handle_simplex with len=3.
            pass
        elif np.dot(acd_n, ao) > 0:
            simplex.remove(b)
            direction[:] = acd_n
            pass
        elif np.dot(adb_n, ao) > 0:
            simplex.remove(c)
            direction[:] = adb_n
            pass
        else:
            # Inside all faces -> Collision!
            return True
            
    return False

def epa(shape1_verts, shape2_verts, simplex):
    """
    Expanding Polytope Algorithm to find penetration depth and normal.
    :return: depth, normal (on surface of B pointing out?)
    """
    # Ensure simplex is a tetrahedron
    simplex = list(simplex) # Copy
    
    # If standard GJK returned a simplex < 4 points but claimed collision,
    # it implies the origin is very close to a lower-dimensional feature.
    # We try to expand it to 4 points to run EPA.
    if len(simplex) < 2:
         return 0.0, np.array([1.0, 0, 0]) # Degenerate

    while len(simplex) < 4:
         if len(simplex) == 2:
             ab = simplex[1] - simplex[0]
             # Try a perpendicular direction
             d = np.cross(ab, np.array([1.0, 0, 0]))
             if np.linalg.norm(d) < 1e-6: d = np.cross(ab, np.array([0, 1.0, 0]))
         elif len(simplex) == 3:
             ab = simplex[1] - simplex[0]
             ac = simplex[2] - simplex[0]
             d = np.cross(ab, ac)
             if np.linalg.norm(d) < 1e-6: d = np.array([1.0, 0, 0]) # Colinear triangle?
             
         d = d / (np.linalg.norm(d) + 1e-10)
         
         # Try +d
         p = get_support(shape1_verts, shape2_verts, d)
         
         # Ensure p is distinct
         if any(np.linalg.norm(p - s) < 1e-5 for s in simplex):
              p = get_support(shape1_verts, shape2_verts, -d)
              
         if any(np.linalg.norm(p - s) < 1e-5 for s in simplex):
              # Cannot expand, flat volume
              return 0.0, d 
         
         simplex.append(p)

    # 1. Construct initial polytope (Tetrahedron) from simplex
    # Faces are list of indices or points.
    faces = [
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
        [1, 3, 2] 
    ]
    
    polytope = list(simplex) 
    
    # helper to compute normal and distance of a face
    def get_face_dist_norm(face_indices):
        v1 = polytope[face_indices[0]]
        v2 = polytope[face_indices[1]]
        v3 = polytope[face_indices[2]]
        n = np.cross(v2 - v1, v3 - v1)
        norm_len = np.linalg.norm(n)
        if norm_len < 1e-10: return np.inf, n # Degenerate
        n = n / norm_len
        dist = np.dot(n, v1)
        
        # Ensure normal points away from origin
        if dist < 0:
            dist = -dist
            n = -n
            # Fix winding in 'faces' if we were modifying it, but here we just need n/dist
            
        return dist, n

    min_face_idx = -1
    min_dist = np.inf
    min_norm = None
    
    TOLERANCE = 1e-4
    MAX_ITER = 60
    
    for _ in range(MAX_ITER):
        # Find closest face
        min_dist = np.inf
        min_face_idx = -1
        min_norm = None
        
        for i, f in enumerate(faces):
            d, n = get_face_dist_norm(f)
            if d < min_dist:
                min_dist = d
                min_norm = n
                min_face_idx = i
                
        # Search support in direction of normal
        # We want the point on Minkowski boundary furthest in normal dist
        s_point = get_support(shape1_verts, shape2_verts, min_norm)
        
        s_dist = np.dot(s_point, min_norm)
        
        if abs(s_dist - min_dist) < TOLERANCE:
            return min_dist, min_norm
            
        # Expand polytope
        # Remove faces visible from s_point (forming a "horizon")
        
        # Helper: is visible?
        # dot(normal, s_point - face_point) > 0 implies s_point is "above" face
        # but face_dist is dot(n, v1). simpler: dot(n, s_point) > face_dist + tol
        
        new_faces = []
        horizon_edges = [] 
        
        # Check visibility for all faces
        # Note: Optimization - usually we start BFS from min_face, but rigorous way loops all
        visible_faces = []
        for i, f in enumerate(faces):
            d, n = get_face_dist_norm(f)
            if np.dot(n, s_point) > d + TOLERANCE:
                visible_faces.append(i)
            else:
                new_faces.append(f)
        
        # Find horizon edges: edges shared by 1 visible and 1 non-visible face
        # But since we iterate, we can just grab all edges of visible faces, 
        # and keep those that are unique (count=1 in the visible set) ? 
        # Standard EPA: The horizon is the boundary of the visible region.
        
        # Collect edges of visible faces
        vis_edges = []
        for idx in visible_faces:
             f = faces[idx]
             vis_edges.append((f[0], f[1]))
             vis_edges.append((f[1], f[2]))
             vis_edges.append((f[2], f[0]))
             
        # Filter for unique edges locally to the visible set?
        # An edge is internal to the visible cap if it appears twice (A->B and B->A in different faces).
        # An edge is on the horizon if it appears once.
        # We need canonical edges to count. Sorted tuple? 
        # Winding matters for new faces.
        # Edge (A, B) in visible face means we need new face (A, B, S_point).
        # If (B, A) also exists in another visible face, it's internal.
        
        from collections import defaultdict
        edge_count = defaultdict(int)
        for u, v in vis_edges:
            edge_count[(u, v)] += 1
            
        # Standard EPA horizon logic
        # If an edge (u, v) is in a visible face, check (v, u). 
        # If (v, u) is NOT in a visible face, then (u, v) is a horizon edge.
        # This requires looking up if neighbor is visible.
        
        # Simpler: Count occurrences of sorted edges? No, direction matters.
        # If using exact winding: unique edges in `vis_edges` list are horizon?
        # (u,v) from f1, (v,u) from f2. If both f1,f2 visible, both edges added.
        # We can remove pairs (u,v) and (v,u).
        
        unique_edges = []
        # Naive removal of pairs
        temp_edges = vis_edges.copy()
        while temp_edges:
            e = temp_edges.pop(0)
            reverse_e = (e[1], e[0])
            if reverse_e in temp_edges:
                temp_edges.remove(reverse_e)
            else:
                unique_edges.append(e)
                
        # Add new faces connecting horizon to s_point
        polytope.append(s_point)
        s_idx = len(polytope) - 1
        
        for u, v in unique_edges:
            new_faces.append([u, v, s_idx])
            
        faces = new_faces

    return min_dist, min_norm

