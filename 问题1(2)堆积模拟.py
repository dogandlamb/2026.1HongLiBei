import sys
sys.path.append('src')
import numpy as np
import collision_detection_custom as cd
import particle_generator
import voronoi_generator
import size_distribution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
import trimesh

class Particle:
    def __init__(self, id, vertices, density=2.5):
        self.id = id
        # Local vertices centered at (0,0,0)
        self.local_vertices = np.array(vertices)
        
        # Calculate properties
        self.mass = len(vertices) * density # Simplified mass proportional to vertex count ~ volume
        # Center the vertices
        center = np.mean(self.local_vertices, axis=0)
        self.local_vertices -= center
        
        # In a real DEM, we calculate Moment of Inertia Tensor I_body
        # Simplified: Identity scaled by mass
        self.inertia = np.eye(3) * self.mass * 0.1 
        self.inv_inertia = np.linalg.inv(self.inertia)
        self.inv_mass = 1.0 / self.mass
        
        # State
        self.position = np.array([0.0, 0.0, 0.0])
        self.rotation = np.eye(3) # Orientation Matrix
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        self.force = np.array([0.0, 0.0, 0.0])
        self.torque = np.array([0.0, 0.0, 0.0])
        
    def get_world_vertices(self):
        # Apply rotation and translation
        return np.dot(self.local_vertices, self.rotation.T) + self.position
        
    def integrate(self, dt):
        # Safety clamp for forces
        max_force = 1e5
        force_norm = np.linalg.norm(self.force)
        if force_norm > max_force:
            self.force *= (max_force / force_norm)
            
        torque_norm = np.linalg.norm(self.torque)
        if torque_norm > max_force:
            self.torque *= (max_force / torque_norm)

        # Symplectic Euler / Semi-implicit
        acc = self.force * self.inv_mass
        self.velocity += acc * dt
        self.position += self.velocity * dt
        
        # Angular
        ang_acc = np.dot(self.inv_inertia, self.torque) # Simplified (no gyroscopic)
        self.angular_velocity += ang_acc * dt
        
        # Damping (Global artificial damping to stabilize simulation)
        self.velocity *= 0.999
        self.angular_velocity *= 0.999
        
        # Update Rotation: R_new = R_old + skew(w) * R * dt
        # Or using axis angle. Small angle approx:
        theta = np.linalg.norm(self.angular_velocity) * dt
        
        # Check for NaN/Inf
        if not np.isfinite(theta):
            self.angular_velocity[:] = 0
            theta = 0
            
        if theta > 1e-6:
            axis = self.angular_velocity / (theta/dt)
            # Rodrigues rotation formula
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R_inc = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*np.dot(K, K)
            self.rotation = np.dot(R_inc, self.rotation)
            
            # Re-orthogonalize to prevent drift
            try:
                u, _, vt = np.linalg.svd(self.rotation)
                self.rotation = np.dot(u, vt)
            except np.linalg.LinAlgError:
                # Reset if fails
                self.rotation = np.eye(3)
                self.angular_velocity[:] = 0

        # Reset forces
        self.force[:] = 0
        self.torque[:] = 0

class Container:
    def __init__(self, radius, height):
        self.radius = radius
        self.height = height
        
    def check_collision(self, particle):
        # floor = 0, walls cylinder radius
        pts = particle.get_world_vertices()
        
        # Floor (z=0)
        min_z = np.min(pts[:, 2])
        if min_z < 0:
            # Simple penalty
            depth = -min_z
            kn = 1e4
            gn = 100
            
            # Just apply upward force at centroid for stability (Simplified for wall contact)
            # Correct: Apply at lowest points
            # For simplicity in this demo: Apply to Center of Mass
            normal = np.array([0, 0, 1])
            
            v_normal = np.dot(particle.velocity, normal)
            f_mag = kn * depth - gn * v_normal
            if f_mag < 0: f_mag = 0
            
            particle.force += f_mag * normal
            
        # Cylinder Walls
        # Check radius of points
        radii = np.sqrt(pts[:,0]**2 + pts[:,1]**2)
        max_r_idx = np.argmax(radii)
        max_r = radii[max_r_idx]
        
        if max_r > self.radius:
            depth = max_r - self.radius
            pt = pts[max_r_idx]
            
            # Normal points INWARD (towards center)
            normal = -np.array([pt[0], pt[1], 0]) 
            n_norm = np.linalg.norm(normal)
            if n_norm > 0: normal /= n_norm
            
            kn = 1e4
            gn = 100
            
            # Point velocity
            # v_pt = v_cm + w x r
            r_vec = pt - particle.position
            v_pt = particle.velocity + np.cross(particle.angular_velocity, r_vec)
            
            v_rel = np.dot(v_pt, normal)
            
            f_mag = kn * depth - gn * v_rel
            if f_mag < 0: f_mag = 0
            
            f_vec = f_mag * normal
            particle.force += f_vec
            particle.torque += np.cross(r_vec, f_vec)

def run_simulation():
    # 1. Generate Particles (using Visualization logic setup)
    # Container Radius = 500 um. Generation area should act as a hopper above it.
    space_bounds = ((0, 1200), (0, 1200), (0, 1200)) 
    density = 5e-6 # Adjusted density for reasonable count in volume
    seeds = voronoi_generator.generate_poisson_seeds(space_bounds, density, random_seed=42)
    polyhedrons = voronoi_generator.generate_voronoi_polyhedrons(seeds)
    
    particles = []
    # Take valid polyhedrons
    count = 0
    keys = list(polyhedrons.keys())
    
    # Position them in a grid above the container
    # Container R=500, so x,y in [-300, 300] to fall inside
    
    grid_w = 6 # 6x6 grid per layer
    spacing = 150 # Spacing between particles (max D ~ 90, so 150 is safe)
    
    for i, k in enumerate(keys):
        if count >= 100: break # Limit
        
        verts = polyhedrons[k]
        
        # Scale logic
        vol = 0
        try:
            # Calculate volume using ConvexHull
            from scipy.spatial import ConvexHull
            hull = ConvexHull(verts)
            vol = hull.volume
        except Exception as e:
            print(f"Skipping particle {i}: Geometry error {e}")
            continue
            
        # Use log-normal parameters as per paper: ln(D) ~ N(4.0, 0.3^2)
        # This gives median D = exp(4.0) ~= 54.6 um, covering 30-90 range well.
        target_d = size_distribution.generate_normal_particle_size(4.0, 0.3, (30, 90))
        current_d = (6 * vol / np.pi)**(1/3)
        if current_d == 0: continue
        
        scale_factor = target_d / current_d
        verts = [ (v[0]*scale_factor, v[1]*scale_factor, v[2]*scale_factor) for v in verts]

        # Modify with noise
        verts = particle_generator.perlin_noise_modification(verts, 2.5)
        
        # Create Particle with normalized mass for stability
        # Virtual Mass = 1.0
        p = Particle(i, verts, density=1.0) # Density irrelevant if we overwrite mass
        p.mass = 1.0
        p.inv_mass = 1.0
        p.inertia = np.eye(3) * 100.0 # Virtual Inertia
        p.inv_inertia = np.linalg.inv(p.inertia)
        
        # Grid placement
        ix = i % grid_w
        iy = (i // grid_w) % grid_w
        iz = i // (grid_w * grid_w)
        
        p.position = np.array([
            (ix - grid_w/2 + 0.5) * spacing, 
            (iy - grid_w/2 + 0.5) * spacing, 
            500.0 + iz * spacing * 1.5 
        ])
        
        particles.append(p)
        count += 1
        
    print(f"Generated {len(particles)} particles.")
        
    container = Container(radius=500.0, height=2000.0)
    
    # Simulation Loop
    dt = 0.005 # Stable time step
    total_time = 3.0 
    steps = int(total_time / dt)
    
    # Virtual Gravity (tuned for visualization speed)
    gravity = np.array([0, 0, -5000.0]) 
    
    # Virtual Stiffness
    Kn = 1e5 
    Gn = 500 
    
    print(f"Starting simulation with {len(particles)} particles for {steps} steps...")
    
    for step in range(steps):
        if step % 100 == 0: 
            # Debug info
            z_positions = [p.position[2] for p in particles]
            avg_z = np.mean(z_positions) if z_positions else 0
            min_z = np.min(z_positions) if z_positions else 0
            print(f"Step {step}/{steps} - Avg Z: {avg_z:.1f}, Min Z: {min_z:.1f}")
    
    for step in range(steps):
        if step % 50 == 0: print(f"Step {step}/{steps}")
        
        # 1. Apply Gravity
        for p in particles:
            p.force += gravity * p.mass
            
        # 2. Wall Collisions
        for p in particles:
            container.check_collision(p)
            
        # 3. Particle-Particle Collisions (Brute Force N^2)
        for i in range(len(particles)):
            for j in range(i+1, len(particles)):
                p1 = particles[i]
                p2 = particles[j]
                
                # Broad phase: Bounding Sphere
                dist = np.linalg.norm(p1.position - p2.position)
                # Max radius approx?
                r1 = np.max(np.linalg.norm(p1.local_vertices, axis=1))
                r2 = np.max(np.linalg.norm(p2.local_vertices, axis=1))
                
                if dist < (r1 + r2):
                    # Narrow phase: GJK
                    pts1 = p1.get_world_vertices()
                    pts2 = p2.get_world_vertices()
                    
                    is_colliding, simplex = cd.gjk(pts1, pts2)
                    
                    if is_colliding:
                        # EPA
                        # Warning: EPA is expensive and can fail for deep penetrations
                        try:
                            bg = time.time()
                            depth, normal = cd.epa(pts1, pts2, simplex)
                            # Normal points from 2 to 1? Or 1 to 2?
                            # EPA returns normal pointing OUT of Minkowski Diff (A-B) towards Origin?
                            # Usually means B pushes A. So Force on A is along Normal.
                            
                            # Contact Point approx (midway along penetration)
                            # For simplified DEM, we just apply force at CM or better contact point
                            # Application point = difficult without full contact manifold
                            # Assume contact at p1_cm - (something)? 
                            # Let's apply Force at CM + Torque based on estimated contact point?
                            
                            # Simple: Apply Force at Centroids? No, no torque.
                            # Better: Contact point = p1.pos - normal * (dist_to_surface?)
                            # Let's just apply Force and a small Damping.
                            
                            # Relative Velocity
                            v1 = p1.velocity
                            v2 = p2.velocity
                            v_rel = np.dot(v1 - v2, normal)
                            
                            f_mag = Kn * depth - Gn * v_rel
                            if f_mag < 0: f_mag = 0 # No suction
                            
                            f_vec = f_mag * normal
                            
                            p1.force += f_vec
                            p2.force -= f_vec
                            
                            # Friction / Torque (Simplified)
                            # Ideally: F_t = ...
                            
                        except Exception as e:
                            pass # EPA failure fallback
                        
        # 4. Integrate
        for p in particles:
            p.integrate(dt)
            
    return particles, container

def visualize_packing(particles, container):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Draw Container
    # Cylinder frame
    z = np.linspace(0, 10, 2) # Just indicate height
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = container.radius * np.cos(theta_grid)
    y_grid = container.radius * np.sin(theta_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.1, color='gray')
    
    # Bottom circle
    ax.plot(x_grid[0], y_grid[0], z_grid[0], color='k')
    
    # 2. Draw Particles
    colors = plt.cm.jet(np.linspace(0, 1, len(particles)))
    
    for i, p in enumerate(particles):
        verts = p.get_world_vertices()
        
        # Draw Faces
        # We need hull faces. Since topology doesn't change, we could store it.
        # But hull is fast for display.
        from scipy.spatial import ConvexHull
        hull = ConvexHull(verts)
        faces = [verts[s] for s in hull.simplices]
        
        poly = Poly3DCollection(faces, alpha=0.8, edgecolor='k', facecolor=colors[i])
        ax.add_collection3d(poly)
        
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_zlabel('Z (um)')
    ax.set_xlim(-container.radius-100, container.radius+100)
    ax.set_ylim(-container.radius-100, container.radius+100)
    ax.set_zlim(0, 1500)
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Random Packing in Cylindrical Container (DEM+GJK)")
    
    plt.show()

def save_to_glb(particles, container, filename="packing_simulation.glb"):
    print("Generating GLB file...")
    scene = trimesh.Scene()
    
    # 1. Container (Cylinder)
    # trimesh cylinder is centered at (0,0,0) with height along Z
    cyl = trimesh.creation.cylinder(radius=container.radius, height=container.height, sections=64)
    # Shift so bottom is at Z=0
    cyl.apply_translation([0, 0, container.height/2])
    # Transparent gray
    cyl.visual.face_colors = [200, 200, 200, 50]
    scene.add_geometry(cyl)
    
    # 2. Particles
    from scipy.spatial import ConvexHull
    for i, p in enumerate(particles):
        verts = p.get_world_vertices()
        try:
            # Create a Convex Hull mesh from vertices using scipy to get faces
            hull = ConvexHull(verts)
            mesh = trimesh.Trimesh(vertices=verts, faces=hull.simplices)
            
            # Assign random color
            mesh.visual.face_colors = trimesh.visual.random_color()
            scene.add_geometry(mesh)
        except Exception as e:
            print(f"Error meshing particle {p.id}: {e}")
            
    scene.export(filename)
    print(f"GLB file saved to: {filename}")

if __name__ == "__main__":
    final_particles, container = run_simulation()
    
    # Export GLB first
    save_to_glb(final_particles, container, "outputs/output.glb")
    
    visualize_packing(final_particles, container)
