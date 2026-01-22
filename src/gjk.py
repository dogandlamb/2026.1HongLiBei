import numpy as np
import sys

# Small epsilon for numerical stability
EPSILON = 1e-6


def support(poly1, poly2, direction):
    """
    Support function for Minkowski difference (A - B).
    Returns the point in (poly1 - poly2) furthest in direction d.
    """
    # Find point in poly1 furthest in direction
    idx1 = np.argmax(np.dot(poly1, direction))
    p1 = poly1[idx1]

    # Find point in poly2 furthest in -direction
    idx2 = np.argmax(np.dot(poly2, -direction))
    p2 = poly2[idx2]

    return p1 - p2


def gjk(poly1, poly2):
    """
    Gilbert-Johnson-Keerthi collision detection.
    Returns (is_colliding, simplex).
    """
    poly1 = np.asarray(poly1, dtype=float)
    poly2 = np.asarray(poly2, dtype=float)

    # Initial search direction (arbitrary)
    direction = np.array([1.0, 0.0, 0.0])
    
    # First point
    a = support(poly1, poly2, direction)
    simplex = [a]
    
    # Next direction towards origin
    direction = -a

    for _ in range(64):  # Max iterations
        if np.linalg.norm(direction) < EPSILON:
             # Origin captured or very close
             return True, simplex
             
        a = support(poly1, poly2, direction)
        
        if np.dot(a, direction) < 0:
            # Did not pass the origin, so no collision
            return False, simplex
            
        simplex.append(a)
        
        if _do_simplex(simplex, direction):
            return True, simplex
            
    return False, simplex


def _do_simplex(simplex, direction):
    """
    Evolves the simplex.
    Returns True if origin is contained.
    Updates simplex and direction in-place.
    """
    # Depending on simplex size (line, triangle, tetrahedron)
    n = len(simplex)
    
    if n == 2: # Line
        # A is last added
        a = simplex[-1]
        b = simplex[-0] # Previous
        
        ab = b - a
        ao = -a
        
        if np.dot(ab, ao) > 0:
            # Origin is in region of AB
            direction[:] = np.cross(np.cross(ab, ao), ab)
        else:
            # Origin is in region of A, but we just came from direction of origin
            # This should technically not happen in GJK given the search direction,
            # but for safety:
            simplex[:] = [a]
            direction[:] = ao
            
        return False
        
    elif n == 3: # Triangle
        a = simplex[-1]
        b = simplex[-2]
        c = simplex[-3]
        
        ab = b - a
        ac = c - a
        ao = -a
        
        abc = np.cross(ab, ac)
        
        # Check if origin is above triangle (in direction of normal)
        # We need to make sure normal points away from origin or determining regions
        
        # Edge AB
        edge_normal_ab = np.cross(ab, abc)
        
        # Edge AC
        edge_normal_ac = np.cross(abc, ac)
        
        if np.dot(edge_normal_ab, ao) > 0:
            if np.dot(ab, ao) > 0:
                # Region AB
                simplex[:] = [b, a] # Keep [b, a] ? Or just [a, b]
                direction[:] = np.cross(np.cross(ab, ao), ab)
            else:
                # Region A (AC check? No, just restart)
                if np.dot(ac, ao) > 0:
                     simplex[:] = [c, a]
                     direction[:] = np.cross(np.cross(ac, ao), ac)
                else:    
                     simplex[:] = [a]
                     direction[:] = ao
        elif np.dot(edge_normal_ac, ao) > 0:
            if np.dot(ac, ao) > 0:
                # Region AC
                simplex[:] = [c, a]
                direction[:] = np.cross(np.cross(ac, ao), ac)
            else:
                # Region A
                simplex[:] = [a]
                direction[:] = ao
        else:
            if np.dot(abc, ao) > 0:
                # Region ABC (top)
                direction[:] = abc
            else:
                # Region ABC (bottom) -- invert face winding or normal
                # To ensure consistency
                simplex[:] = [a, c, b] 
                direction[:] = -abc
                
        return False
        
    elif n == 4: # Tetrahedron
        a = simplex[-1]
        b = simplex[-2]
        c = simplex[-3]
        d = simplex[-4]
        
        ao = -a
        ab = b - a
        ac = c - a
        ad = d - a
        
        # Normals of faces (pointing out?)
        # We constructed such that 'abc' was pointing towards origin previously?
        # Actually GJK logic for tetra usually checks face normals
        
        abc = np.cross(ab, ac)
        acd = np.cross(ac, ad)
        adb = np.cross(ad, ab)
        
        # We know origin is 'below' bcd (since d was processed before)
        # So we only check new faces connected to a
        
        # Check direction of normal to ensure it points OUTSIDE the tetrahedron
        # Compare with AD for ABC
        if np.dot(abc, ad) > 0: abc = -abc
        if np.dot(acd, ab) > 0: acd = -acd
        if np.dot(adb, ac) > 0: adb = -adb
        
        if np.dot(abc, ao) > 0:
            simplex[:] = [a, b, c]
            direction[:] = abc
        elif np.dot(acd, ao) > 0:
            simplex[:] = [a, c, d]
            direction[:] = acd
        elif np.dot(adb, ao) > 0:
            simplex[:] = [a, d, b]
            direction[:] = adb
        else:
            return True # Origin inside all faces!
            
        return False

    return False

def epa(poly1, poly2, simplex, tolerance=1e-5):
    """
    Expanding Polytope Algorithm.
    Returns (depth, normal).
    Normal points from poly2 to poly1 (direction to push poly1 to resolve).
    """
    poly1 = np.asarray(poly1, dtype=float)
    poly2 = np.asarray(poly2, dtype=float)

    # Ensure simplex is a tetrahedron
    # GJK might return a simplex of size 1, 2, 3 or 4.
    # If 4, we use it directly.
    # If < 4, it means we touched logic where origin is on boundary or numerical issues.
    # But usually GJK returns true only if enclosed.
    # However, sometimes we might graze.
    
    # We will assume simplex is robust enough or we reconstruct.
    # For robust physics, usually we force a tetrahedron if we are seemingly coplanar.
    
    faces = []
    
    if len(simplex) == 4:
        faces = [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2] # Ensure winding
        ]
        # Clean winding so normals point away from geometric center / origin
        # Note: Origin is inside.
    else:
        # Fallback or error, assume shallow penetration
        return 0.0, np.array([0.0, 0.0, 1.0])
        
    # Helper to get normal
    def get_face_dist_normal(face, polytope):
        a = polytope[face[0]]
        b = polytope[face[1]]
        c = polytope[face[2]]
        
        n = np.cross(b - a, c - a)
        l = np.linalg.norm(n)
        if l < 1e-10:
             return np.inf, n # Degenerate
        n /= l
        
        dist = np.dot(n, a)
        
        # Enforce normal points away from origin
        if dist < 0:
            dist = -dist
            n = -n
            # Fix winding in face for future iterations? Not strictly needed if we just use normal
            
        return dist, n

    polytope = list(simplex) # List of points
    
    # Check winding for initial faces
    # If origin is inside, dot(normal, vertex) > 0
    # We made dist positive above, so normal points OUT.
    
    for _ in range(64): # Max EPA iterations
        closest_face = None
        min_dist = np.inf
        best_normal = None
        
        # Find closest face to origin
        for i, f in enumerate(faces):
            d, n = get_face_dist_normal(f, polytope)
            if d < min_dist:
                min_dist = d
                closest_face = i
                best_normal = n
                
        if closest_face is None:
            break
            
        # Search for support point in direction of normal
        p = support(poly1, poly2, best_normal)
        
        d_p = np.dot(p, best_normal)
        
        if d_p - min_dist < tolerance:
            # Convergence
            if abs(min_dist) < 1e-9 and np.linalg.norm(best_normal) < 1e-9:
                return 0.0, np.array([0.0,0.0,1.0])
            return d_p, best_normal
            
        # Add new point
        faces = _expand_polytope(polytope, faces, p, closest_face)
        
    return min_dist, best_normal

def _expand_polytope(polytope, faces, new_point, old_face_index):
    # Horizon algorithm
    # Find all faces visible from new_point
    
    new_idx = len(polytope)
    polytope.append(new_point)
    
    # We definitely remove the face that was closest
    # And any neighbors that are also visible
    
    # Use a simple flood fill or iteration since N is small
    visible_faces = []
    
    # Check visibility for all faces (naive but robust for small polytopes)
    for i, f in enumerate(faces):
        a = polytope[f[0]]
        b = polytope[f[1]]
        c = polytope[f[2]]
        
        n = np.cross(b - a, c - a)
        n /= (np.linalg.norm(n) + 1e-12)
        
        # Check alignment with origin
        d = np.dot(n, a)
        if d < 0: 
            n = -n
            d = -d
            
        # Check if new_point is in front
        # dist = dot(n, p) - d
        if np.dot(n, new_point) - d > 1e-6:
             visible_faces.append(i)
             
    # Find horizon edges
    # Edges that are shared between a visible and an invisible face
    edges = {}
    
    for fi in visible_faces:
        f = faces[fi]
        # 3 edges: (0,1), (1,2), (2,0)
        es = [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]
        for v1, v2 in es:
            # Sort for unique key
            key = tuple(sorted((v1, v2)))
            if key in edges:
                edges[key] += 1
            else:
                edges[key] = 1
                
    # Keep edges with count 1 (horizon)
    horizon_edges = []
    for k, count in edges.items():
        if count == 1:
            # We need the ordered edge from the face for consistent winding?
            # actually we just connect new_point to these vertices.
            # We need to find the vertex order that matches the adjacent invisible face.
            # But here we lost the info of which visible face it came from.
            # However, since we reconstructed the hull, we might just need to fetch the edge
            # from one of the visible faces and verify if it's the horizon.
            horizon_edges.append(k)

    # Remove visible faces
    new_faces = []
    visible_set = set(visible_faces)
    for i, f in enumerate(faces):
         if i not in visible_set:
             new_faces.append(f)
             
    # Add new faces
    # Naive winding: just append. 
    # Real EPA implementation requires maintaining winding order carefully.
    # The edges in horizon_edges are (u,v). The new triangle is (u,v, new_point).
    # We need to check the normal of (u, v, new_point) vs origin
    for v1, v2 in horizon_edges:
        # Construct triangle
        tri = [v1, v2, new_idx]
        
        # Check normal
        a = polytope[v1]
        b = polytope[v2]
        c = polytope[new_idx] # new point
        
        n = np.cross(b - a, c - a)
        if np.dot(n, a) < 0: # Points towards origin
             n = -n
             # Flip winding? Or just rely on get_face_dist_normal correcting it?
             # get_face_dist_normal corrects the normal direction regardless of winding.
             pass
             
        new_faces.append(tri)
        
    return new_faces
