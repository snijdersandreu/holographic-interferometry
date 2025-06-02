import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def create_cube_surface_full():
    """Create a complete 3D cube surface with ultra-high resolution"""
    size = 1.0
    n_points = 300  # Increased back to 300 for finer detail
    
    # Create all 6 faces of the cube
    faces_data = []
    
    # Define face coordinates
    coords = np.linspace(-size/2, size/2, n_points)
    U, V = np.meshgrid(coords, coords)
    
    # Front face (z = +size/2)
    X_front = U
    Y_front = V  
    Z_front = np.ones_like(U) * (size/2)
    faces_data.append((X_front, Y_front, Z_front, 'front'))
    
    # Back face (z = -size/2) 
    X_back = U
    Y_back = V
    Z_back = np.ones_like(U) * (-size/2)
    faces_data.append((X_back, Y_back, Z_back, 'back'))
    
    # Right face (x = +size/2)
    X_right = np.ones_like(U) * (size/2)
    Y_right = V
    Z_right = U
    faces_data.append((X_right, Y_right, Z_right, 'right'))
    
    # Left face (x = -size/2)
    X_left = np.ones_like(U) * (-size/2)
    Y_left = V
    Z_left = U
    faces_data.append((X_left, Y_left, Z_left, 'left'))
    
    # Top face (y = +size/2)
    X_top = U
    Y_top = np.ones_like(U) * (size/2)
    Z_top = V
    faces_data.append((X_top, Y_top, Z_top, 'top'))
    
    # Bottom face (y = -size/2)
    X_bottom = U
    Y_bottom = np.ones_like(U) * (-size/2)
    Z_bottom = V
    faces_data.append((X_bottom, Y_bottom, Z_bottom, 'bottom'))
    
    return faces_data

def simulate_interferometer_measurement_full(faces_data, measurement_direction='front'):
    """Simulate interferometer measurements on the full cube"""
    
    if measurement_direction == 'front':
        reference_point = np.array([0, 0, 3])
        # Only measure faces that are visible from the front
        visible_faces = ['front', 'right', 'left', 'top', 'bottom']
    elif measurement_direction == 'side':
        reference_point = np.array([3, 0, 0])
        visible_faces = ['right', 'front', 'back', 'top', 'bottom']
    elif measurement_direction == 'top':
        reference_point = np.array([0, 3, 0])
        visible_faces = ['top', 'front', 'back', 'right', 'left']
    
    # Combine all visible faces
    X_combined = []
    Y_combined = []
    Z_combined = []
    distances_combined = []
    
    for X_face, Y_face, Z_face, face_name in faces_data:
        if face_name in visible_faces:
            # Calculate distances from reference point
            distances = np.sqrt((X_face - reference_point[0])**2 + 
                              (Y_face - reference_point[1])**2 + 
                              (Z_face - reference_point[2])**2)
            
            X_combined.append(X_face)
            Y_combined.append(Y_face)
            Z_combined.append(Z_face)
            distances_combined.append(distances)
    
    # Stack all visible faces
    X_all = np.vstack(X_combined)
    Y_all = np.vstack(Y_combined)  
    Z_all = np.vstack(Z_combined)
    distances_all = np.vstack(distances_combined)
    
    return X_all, Y_all, Z_all, distances_all, reference_point

def create_interference_from_distances(distances, wavelength=5e-3, reference_distance=3.0):
    """Create interference pattern from distance measurements"""
    # Calculate path difference
    path_diff = distances - reference_distance
    
    # Phase difference
    phase = 2 * np.pi * path_diff / wavelength
    
    # Interference intensity
    intensity = 1 + np.cos(phase)
    
    return intensity, phase, path_diff

def get_projection_for_viewing(X, Y, Z, direction):
    """Get appropriate 2D projection for viewing direction"""
    if direction == 'front':
        # Project onto X-Y plane (viewing from Z direction)
        return X, Y
    elif direction == 'side':
        # Project onto Z-Y plane (viewing from X direction)  
        return Z, Y
    elif direction == 'top':
        # Project onto X-Z plane (viewing from Y direction)
        return X, Z

def plot_cube_interferometer_holography():
    """Show how interferometer measurements from different angles work with a cube"""
    fig = plt.figure(figsize=(20, 12))
    
    # Create full cube
    faces_data = create_cube_surface_full()
    
    # 1. 3D Cube with measurement directions (large subplot)
    ax1 = fig.add_subplot(2, 4, (1, 2))
    ax1 = fig.add_axes([0.05, 0.55, 0.4, 0.4], projection='3d')
    
    # Plot the cube wireframe for visualization
    size = 0.5
    vertices = [
        [-size, -size, -size], [size, -size, -size], [size, size, -size], [-size, size, -size],
        [-size, -size, size], [size, -size, size], [size, size, size], [-size, size, size]
    ]
    
    cube_faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]], 
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[4], vertices[7], vertices[3], vertices[0]]
    ]
    
    cube = Poly3DCollection(cube_faces, alpha=0.3, facecolor='lightblue', edgecolor='black', linewidth=2)
    ax1.add_collection3d(cube)
    
    # Add measurement directions with arrows
    arrow_length = 1.5
    
    # Front measurement
    ax1.quiver(0, 0, 3, 0, 0, -arrow_length, color='red', arrow_length_ratio=0.1, linewidth=3)
    ax1.text(0, 0, 3.5, 'Front\nMeasurement', ha='center', fontsize=10, color='red', fontweight='bold')
    
    # Side measurement  
    ax1.quiver(3, 0, 0, -arrow_length, 0, 0, color='blue', arrow_length_ratio=0.1, linewidth=3)
    ax1.text(3.5, 0, 0, 'Side\nMeasurement', ha='center', fontsize=10, color='blue', fontweight='bold')
    
    # Top measurement
    ax1.quiver(0, 3, 0, 0, -arrow_length, 0, color='green', arrow_length_ratio=0.1, linewidth=3)
    ax1.text(0, 3.5, 0, 'Top\nMeasurement', ha='center', fontsize=10, color='green', fontweight='bold')
    
    ax1.set_title('3D Cube with Interferometer\nMeasurement Directions', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.view_init(elev=20, azim=45)
    
    ax1.set_xlim([-2, 4])
    ax1.set_ylim([-2, 4])
    ax1.set_zlim([-2, 4])
    
    ax1.grid(False)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    
    # 2-4. Distance measurements from different directions
    directions = ['front', 'side', 'top']
    colors = ['Reds', 'Blues', 'Greens']
    titles = ['Front View\n(Z-direction)', 'Side View\n(X-direction)', 'Top View\n(Y-direction)']
    
    for i, (direction, cmap, title) in enumerate(zip(directions, colors, titles)):
        ax = fig.add_subplot(2, 4, 3 + i)
        
        X_all, Y_all, Z_all, distances_all, ref_point = simulate_interferometer_measurement_full(faces_data, direction)
        
        # Get projection coordinates
        x_proj, y_proj = get_projection_for_viewing(X_all, Y_all, Z_all, direction)
        
        # Create scatter plot for better visualization of individual points
        scatter = ax.scatter(x_proj, y_proj, c=distances_all, cmap=cmap, s=0.1, alpha=0.9)
        ax.set_title(f'{title}\nDistance Measurement', fontsize=11, fontweight='bold')
        
        if direction == 'front':
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
        elif direction == 'side':
            ax.set_xlabel('Z (m)')
            ax.set_ylabel('Y (m)')
        elif direction == 'top':
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
            
        plt.colorbar(scatter, ax=ax, shrink=0.7, label='Distance (m)')
        ax.set_aspect('equal')
    
    # 5-7. Corresponding interference patterns
    for i, (direction, cmap, title) in enumerate(zip(directions, colors, titles)):
        ax = fig.add_subplot(2, 4, 6 + i)
        
        X_all, Y_all, Z_all, distances_all, ref_point = simulate_interferometer_measurement_full(faces_data, direction)
        intensity, phase, path_diff = create_interference_from_distances(distances_all)
        
        # Get projection coordinates
        x_proj, y_proj = get_projection_for_viewing(X_all, Y_all, Z_all, direction)
        
        # Create scatter plot of interference pattern
        scatter = ax.scatter(x_proj, y_proj, c=intensity, cmap='gray', s=0.1, alpha=0.9)
        ax.set_title(f'{title}\nInterference Pattern', fontsize=11, fontweight='bold')
        
        if direction == 'front':
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
        elif direction == 'side':
            ax.set_xlabel('Z (m)')
            ax.set_ylabel('Y (m)')
        elif direction == 'top':
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
            
        plt.colorbar(scatter, ax=ax, shrink=0.7, label='Intensity')
        ax.set_aspect('equal')
    
    plt.tight_layout(pad=2.0)
    plt.show()

def plot_cube_holographic_recording():
    """Create the main holographic recording visualization with cube"""
    fig = plt.figure(figsize=(18, 12))
    
    # Create full cube surface
    faces_data = create_cube_surface_full()
    
    # Calculate distances from front view
    X_all, Y_all, Z_all, distances_all, ref_point = simulate_interferometer_measurement_full(faces_data, 'front')
    intensity, phase, path_diff = create_interference_from_distances(distances_all)
    
    # 1. 3D Cube (top left)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    size = 0.5
    vertices = [
        [-size, -size, -size], [size, -size, -size], [size, size, -size], [-size, size, -size],
        [-size, -size, size], [size, -size, size], [size, size, size], [-size, size, size]
    ]
    
    cube_faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[4], vertices[7], vertices[3], vertices[0]]
    ]
    
    cube = Poly3DCollection(cube_faces, alpha=0.4, facecolor='lightblue', edgecolor='black', linewidth=2)
    ax1.add_collection3d(cube)
    
    # Add interferometer setup
    ax1.quiver(0, 0, 3, 0, 0, -1, color='red', arrow_length_ratio=0.1, linewidth=3)
    ax1.text(0, 0, 3.5, 'Interferometer', ha='center', fontsize=10, color='red', fontweight='bold')
    
    ax1.set_title('3D Cube\nwith Interferometer Setup', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('X (m)', fontsize=10)
    ax1.set_ylabel('Y (m)', fontsize=10)
    ax1.set_zlabel('Z (m)', fontsize=10)
    ax1.view_init(elev=25, azim=45)
    
    ax1.set_xlim([-2, 4])
    ax1.set_ylim([-2, 4])
    ax1.set_zlim([-2, 4])
    
    ax1.grid(False)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    
    # 2. Distance map (top center) - using front view projection
    ax2 = fig.add_subplot(2, 3, 2)
    x_proj, y_proj = get_projection_for_viewing(X_all, Y_all, Z_all, 'front')
    scatter2 = ax2.scatter(x_proj, y_proj, c=distances_all, cmap='plasma', s=0.1, alpha=0.9)
    ax2.set_title('Distance Measurements\n(Interferometer detects each point)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (m)', fontsize=10)
    ax2.set_ylabel('Y (m)', fontsize=10)
    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
    cbar2.set_label('Distance (m)', fontsize=10)
    ax2.set_aspect('equal')
    
    # 3. Phase map (top right)
    ax3 = fig.add_subplot(2, 3, 3)
    scatter3 = ax3.scatter(x_proj, y_proj, c=phase, cmap='hsv', s=0.1, alpha=0.9)
    ax3.set_title('Phase Information\n(Encoded in interference)', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('X (m)', fontsize=10)
    ax3.set_ylabel('Y (m)', fontsize=10)
    cbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.8)
    cbar3.set_label('Phase (radians)', fontsize=10)
    ax3.set_aspect('equal')
    
    # 4. Interference pattern (bottom left)
    ax4 = fig.add_subplot(2, 3, 4)
    scatter4 = ax4.scatter(x_proj, y_proj, c=intensity, cmap='gray', s=0.1, alpha=1.0)
    ax4.set_title('Recorded Hologram\n(Interference Pattern)', 
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('X (m)', fontsize=10)
    ax4.set_ylabel('Y (m)', fontsize=10)
    cbar4 = plt.colorbar(scatter4, ax=ax4, shrink=0.8)
    cbar4.set_label('Intensity', fontsize=10)
    ax4.set_aspect('equal')
    
    # 5. Cross-section showing fringes (bottom center)
    ax5 = fig.add_subplot(2, 3, 5)
    
    # Take a slice through the middle for cross-section
    center_mask = np.abs(y_proj) < 0.05  # Points near y=0
    x_slice = x_proj[center_mask]
    intensity_slice = intensity[center_mask]
    distance_slice = distances_all[center_mask]
    
    # Sort by x coordinate for proper line plot
    sort_idx = np.argsort(x_slice)
    x_sorted = x_slice[sort_idx]
    intensity_sorted = intensity_slice[sort_idx]
    distance_sorted = distance_slice[sort_idx]
    
    # Plot both intensity and distance for comparison
    ax5_twin = ax5.twinx()
    line1 = ax5.plot(x_sorted, intensity_sorted, 'b-', linewidth=2, label='Interference fringes')
    line2 = ax5_twin.plot(x_sorted, distance_sorted, 'r--', linewidth=2, label='Object distance')
    
    ax5.set_title('Cross-section: Distance → Fringes', fontsize=12, fontweight='bold')
    ax5.set_xlabel('X position (m)', fontsize=10)
    ax5.set_ylabel('Intensity', fontsize=10, color='blue')
    ax5_twin.set_ylabel('Distance (m)', fontsize=10, color='red')
    ax5.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, loc='upper right', fontsize=9)
    
    # 6. Conceptual summary (bottom right)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.text(0.5, 0.9, 'Holographic Recording Process', 
             ha='center', va='center', fontsize=16, fontweight='bold')
    
    steps = [
        '1. Interferometer measures distance',
        '   to each point on ALL cube faces',
        '2. Distance variations create',
        '   fine interference patterns', 
        '3. Pattern encodes complete 3D info',
        '4. Same principle as interferometry!'
    ]
    
    for i, step in enumerate(steps):
        color = 'red' if 'Same principle' in step else 'black'
        weight = 'bold' if 'Same principle' in step else 'normal'
        ax6.text(0.05, 0.75 - i*0.1, step, 
                ha='left', va='center', fontsize=11, color=color, fontweight=weight)
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    plt.tight_layout(pad=3.0)
    plt.show()

def plot_holographic_reconstruction():
    """Demonstrate the holographic reconstruction process"""
    fig = plt.figure(figsize=(16, 10))
    
    # Create cube and get hologram data
    faces_data = create_cube_surface_full()
    X_all, Y_all, Z_all, distances_all, ref_point = simulate_interferometer_measurement_full(faces_data, 'front')
    intensity, phase, path_diff = create_interference_from_distances(distances_all)
    x_proj, y_proj = get_projection_for_viewing(X_all, Y_all, Z_all, 'front')
    
    # 1. Original 3D Object (top left)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    size = 0.5
    vertices = [
        [-size, -size, -size], [size, -size, -size], [size, size, -size], [-size, size, -size],
        [-size, -size, size], [size, -size, size], [size, size, size], [-size, size, size]
    ]
    cube_faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[4], vertices[7], vertices[3], vertices[0]]
    ]
    cube = Poly3DCollection(cube_faces, alpha=0.7, facecolor='lightblue', edgecolor='black', linewidth=2)
    ax1.add_collection3d(cube)
    ax1.set_title('1. Original Object\n(Cube)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.view_init(elev=25, azim=45)
    ax1.set_xlim([-1, 1]); ax1.set_ylim([-1, 1]); ax1.set_zlim([-1, 1])
    
    # 2. Recorded Hologram (top center)
    ax2 = fig.add_subplot(2, 3, 2)
    scatter2 = ax2.scatter(x_proj, y_proj, c=intensity, cmap='gray', s=0.1, alpha=1.0)
    ax2.set_title('2. Recorded Hologram\n(Interference Pattern)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
    plt.colorbar(scatter2, ax=ax2, shrink=0.7, label='Intensity')
    ax2.set_aspect('equal')
    
    # 3. Reconstruction Beam Setup (top right)
    ax3 = fig.add_subplot(2, 3, 3)
    # Show hologram plane and reconstruction beam
    hologram_x = np.linspace(-0.6, 0.6, 100)
    hologram_y = np.linspace(-0.6, 0.6, 100)
    H_x, H_y = np.meshgrid(hologram_x, hologram_y)
    
    # Reconstruction beam (parallel wavefront)
    ax3.contourf(H_x, H_y, np.ones_like(H_x), levels=1, colors=['lightcyan'], alpha=0.7)
    ax3.contour(H_x, H_y, np.ones_like(H_x), levels=1, colors=['blue'], linewidths=2)
    
    # Add arrows showing beam direction
    for i in range(0, len(hologram_x), 20):
        ax3.arrow(hologram_x[i], 0.8, 0, -0.15, head_width=0.03, head_length=0.02, fc='blue', ec='blue')
    
    ax3.text(0, 0.9, 'Reconstruction Beam', ha='center', fontsize=10, color='blue', fontweight='bold')
    ax3.set_title('3. Reconstruction Setup\n(Beam illuminates hologram)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X (m)'); ax3.set_ylabel('Y (m)')
    ax3.set_xlim(-0.7, 0.7); ax3.set_ylim(-0.7, 1.0)
    ax3.set_aspect('equal')
    
    # 4. Diffraction Pattern (bottom left)
    ax4 = fig.add_subplot(2, 3, 4)
    # Simulate diffraction by showing how fringes scatter light
    diffraction_angles = np.linspace(-np.pi/4, np.pi/4, 50)
    fringe_spacing = np.linspace(0.01, 0.1, 50)  # Different fringe spacings
    ANGLES, SPACING = np.meshgrid(diffraction_angles, fringe_spacing)
    
    # Diffraction efficiency (more efficient for finer fringes)
    diffraction_efficiency = 1.0 / (1 + SPACING * 50)
    
    scatter4 = ax4.contourf(ANGLES * 180/np.pi, SPACING * 1000, diffraction_efficiency, 
                           levels=20, cmap='plasma')
    ax4.set_title('4. Diffraction Pattern\n(Hologram acts as grating)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Diffraction Angle (degrees)')
    ax4.set_ylabel('Fringe Spacing (mm)')
    plt.colorbar(scatter4, ax=ax4, shrink=0.7, label='Efficiency')
    
    # 5. Reconstructed Wavefront (bottom center)
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    
    # Create a "reconstructed" cube (slightly transparent to show it's virtual)
    cube_reconstructed = Poly3DCollection(cube_faces, alpha=0.3, facecolor='red', 
                                        edgecolor='darkred', linewidth=2, linestyle='--')
    ax5.add_collection3d(cube_reconstructed)
    
    # Add some wavefront lines to show reconstruction
    for z_plane in [-0.3, 0, 0.3]:
        circle_theta = np.linspace(0, 2*np.pi, 20)
        circle_x = 0.4 * np.cos(circle_theta)
        circle_y = 0.4 * np.sin(circle_theta)
        circle_z = np.ones_like(circle_x) * z_plane
        ax5.plot(circle_x, circle_y, circle_z, 'r--', alpha=0.7, linewidth=1)
    
    ax5.set_title('5. Reconstructed Object\n(Virtual 3D image)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('X'); ax5.set_ylabel('Y'); ax5.set_zlabel('Z')
    ax5.view_init(elev=25, azim=45)
    ax5.set_xlim([-1, 1]); ax5.set_ylim([-1, 1]); ax5.set_zlim([-1, 1])
    
    # 6. Viewing from Different Angles (bottom right)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.text(0.5, 0.9, 'Holographic Reconstruction', ha='center', va='center', 
             fontsize=14, fontweight='bold')
    
    reconstruction_steps = [
        '• Illuminate hologram with coherent light',
        '• Interference fringes act as diffraction grating',
        '• Light diffracts at specific angles',
        '• Diffracted beams reconstruct original wavefront',
        '• Observer sees 3D virtual image',
        '• Full parallax - different angles show different views!'
    ]
    
    for i, step in enumerate(reconstruction_steps):
        color = 'red' if 'parallax' in step else 'black'
        weight = 'bold' if 'parallax' in step else 'normal'
        ax6.text(0.05, 0.75 - i*0.1, step, ha='left', va='center', 
                fontsize=10, color=color, fontweight=weight)
    
    ax6.set_xlim(0, 1); ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    plt.tight_layout(pad=2.0)
    plt.show()

def create_displaced_cube_surface(displacement_vector=[0.05, 0.02, -0.03], rotation_angle=0.1):
    """Create a cube surface that has been slightly displaced and rotated"""
    size = 1.0
    n_points = 300  # Increased back to 300 for finer detail
    
    faces_data = []
    coords = np.linspace(-size/2, size/2, n_points)
    U, V = np.meshgrid(coords, coords)
    
    # Define face coordinates (same as original cube)
    # Front face (z = +size/2)
    X_front = U
    Y_front = V  
    Z_front = np.ones_like(U) * (size/2)
    faces_data.append((X_front, Y_front, Z_front, 'front'))
    
    # Back face (z = -size/2) 
    X_back = U
    Y_back = V
    Z_back = np.ones_like(U) * (-size/2)
    faces_data.append((X_back, Y_back, Z_back, 'back'))
    
    # Right face (x = +size/2)
    X_right = np.ones_like(U) * (size/2)
    Y_right = V
    Z_right = U
    faces_data.append((X_right, Y_right, Z_right, 'right'))
    
    # Left face (x = -size/2)
    X_left = np.ones_like(U) * (-size/2)
    Y_left = V
    Z_left = U
    faces_data.append((X_left, Y_left, Z_left, 'left'))
    
    # Top face (y = +size/2)
    X_top = U
    Y_top = np.ones_like(U) * (size/2)
    Z_top = V
    faces_data.append((X_top, Y_top, Z_top, 'top'))
    
    # Bottom face (y = -size/2)
    X_bottom = U
    Y_bottom = np.ones_like(U) * (-size/2)
    Z_bottom = V
    faces_data.append((X_bottom, Y_bottom, Z_bottom, 'bottom'))
    
    # Apply displacement and rotation to all faces
    displaced_faces = []
    for X_face, Y_face, Z_face, face_name in faces_data:
        # Apply small rotation around Z-axis
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        
        X_rotated = X_face * cos_theta - Y_face * sin_theta
        Y_rotated = X_face * sin_theta + Y_face * cos_theta
        Z_rotated = Z_face
        
        # Apply displacement
        X_displaced = X_rotated + displacement_vector[0]
        Y_displaced = Y_rotated + displacement_vector[1]
        Z_displaced = Z_rotated + displacement_vector[2]
        
        displaced_faces.append((X_displaced, Y_displaced, Z_displaced, face_name))
    
    return displaced_faces

def plot_double_exposure_holographic_interferometry():
    """Demonstrate the double-exposure method for holographic interferometry"""
    fig = plt.figure(figsize=(18, 11)) # Adjusted for 2x3 + 1 wide bottom row
    
    # Create original and displaced cube
    original_faces = create_cube_surface_full()
    # displacement_vector=[0.05, 0.02, -0.03], rotation_angle=0.1 (around Z)
    displaced_faces_data = create_displaced_cube_surface(displacement_vector=[0.05, 0.02, -0.03], rotation_angle=0.1)
    
    # Calculate holograms for both positions using SIDE view (Y-Z projection)
    X_orig, Y_orig, Z_orig, dist_orig, ref_point_side = simulate_interferometer_measurement_full(original_faces, 'side')
    X_disp, Y_disp, Z_disp, dist_disp, _ = simulate_interferometer_measurement_full(displaced_faces_data, 'side')
    
    # Create interference patterns for both (side view Y-Z)
    intensity_orig_yz, phase_orig_yz, _ = create_interference_from_distances(dist_orig)
    intensity_disp_yz, phase_disp_yz, _ = create_interference_from_distances(dist_disp)
    
    # Get Y-Z projections for side view
    z_proj_yz, y_proj_yz = get_projection_for_viewing(X_orig, Y_orig, Z_orig, 'side')
    
    # Double-exposure interference for Y-Z view
    phase_difference_yz = phase_disp_yz - phase_orig_yz
    double_exposure_intensity_yz = 1 + np.cos(phase_difference_yz)

    # --- Data for X-Z projection (top view) ---
    # Simulate interferometer from TOP for the original and displaced states
    X_orig_top, Y_orig_top, Z_orig_top, dist_orig_top, ref_point_top = simulate_interferometer_measurement_full(original_faces, 'top')
    X_disp_top, Y_disp_top, Z_disp_top, dist_disp_top, _ = simulate_interferometer_measurement_full(displaced_faces_data, 'top')

    # Create interference patterns for top view
    intensity_orig_xz, phase_orig_xz, _ = create_interference_from_distances(dist_orig_top)
    intensity_disp_xz, phase_disp_xz, _ = create_interference_from_distances(dist_disp_top)

    # Get X-Z projections for top view (X from original, Z from original)
    # Note: get_projection_for_viewing returns (X,Z) for 'top' view
    x_proj_xz, z_proj_xz = get_projection_for_viewing(X_orig_top, Y_orig_top, Z_orig_top, 'top')

    # Double-exposure interference for X-Z view
    phase_difference_xz = phase_disp_xz - phase_orig_xz
    double_exposure_intensity_xz = 1 + np.cos(phase_difference_xz)
    # --- End Data for X-Z projection ---

    # 1. Original cube position (3D)
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    size = 0.5
    vertices = [
        [-size, -size, -size], [size, -size, -size], [size, size, -size], [-size, size, -size],
        [-size, -size, size], [size, -size, size], [size, size, size], [-size, size, size]
    ]
    cube_faces_viz = [
        [vertices[i] for i in [0,1,2,3]], [vertices[i] for i in [4,5,6,7]],
        [vertices[i] for i in [0,1,5,4]], [vertices[i] for i in [2,3,7,6]],
        [vertices[i] for i in [1,2,6,5]], [vertices[i] for i in [0,3,7,4]]
    ]
    cube_original_viz = Poly3DCollection(cube_faces_viz, alpha=0.7, facecolor='lightblue', edgecolor='blue', linewidth=1.5)
    ax1.add_collection3d(cube_original_viz)
    ax1.set_title('Original Position (t=0)', fontsize=10, fontweight='bold')
    ax1.set_xlabel('X', fontsize=8); ax1.set_ylabel('Y', fontsize=8); ax1.set_zlabel('Z', fontsize=8)
    ax1.view_init(elev=0, azim=90) # Changed to Y-Z projection (side view)
    ax1.set_xlim([-1,1]); ax1.set_ylim([-1,1]); ax1.set_zlim([-1,1])

    # 2. Displaced cube position (3D)
    ax2 = fig.add_subplot(3, 3, 2, projection='3d')
    disp_vector_viz = [0.05, 0.02, -0.03]
    rotation_angle_viz = 0.1 # Z-axis rotation
    disp_vertices_viz = []
    for v in vertices:
        x_rot = v[0]*np.cos(rotation_angle_viz) - v[1]*np.sin(rotation_angle_viz)
        y_rot = v[0]*np.sin(rotation_angle_viz) + v[1]*np.cos(rotation_angle_viz)
        z_rot = v[2]
        disp_vertices_viz.append([x_rot + disp_vector_viz[0], y_rot + disp_vector_viz[1], z_rot + disp_vector_viz[2]])
    displaced_cube_faces_viz = [
        [disp_vertices_viz[i] for i in [0,1,2,3]], [disp_vertices_viz[i] for i in [4,5,6,7]],
        [disp_vertices_viz[i] for i in [0,1,5,4]], [disp_vertices_viz[i] for i in [2,3,7,6]],
        [disp_vertices_viz[i] for i in [1,2,6,5]], [disp_vertices_viz[i] for i in [0,3,7,4]]
    ]
    cube_displaced_viz = Poly3DCollection(displaced_cube_faces_viz, alpha=0.7, facecolor='lightcoral', edgecolor='red', linewidth=1.5)
    ax2.add_collection3d(cube_displaced_viz)
    ax2.set_title('Displaced Position (t=Δt)', fontsize=10, fontweight='bold')
    ax2.set_xlabel('X', fontsize=8); ax2.set_ylabel('Y', fontsize=8); ax2.set_zlabel('Z', fontsize=8)
    ax2.view_init(elev=0, azim=90) # Changed to Y-Z projection (side view)
    ax2.set_xlim([-1,1]); ax2.set_ylim([-1,1]); ax2.set_zlim([-1,1])

    # 3. Original hologram (Y-Z Projection)
    ax3 = fig.add_subplot(3, 3, 3)
    scatter3 = ax3.scatter(y_proj_yz, z_proj_yz, c=intensity_orig_yz, cmap='gray', s=0.1, alpha=1.0)
    ax3.set_title('Hologram 1 (Y-Z Proj.)', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Y (m)', fontsize=8); ax3.set_ylabel('Z (m)', fontsize=8)
    cbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.7); cbar3.set_label('Intensity', fontsize=8)
    ax3.set_aspect('equal')

    # 4. Displaced hologram (Y-Z Projection)
    ax4 = fig.add_subplot(3, 3, 4)
    scatter4 = ax4.scatter(y_proj_yz, # Use y_proj_yz from original for consistent grid with intensity_disp_yz
                           z_proj_yz, # Use z_proj_yz from original for consistent grid with intensity_disp_yz
                           c=intensity_disp_yz, cmap='gray', s=0.1, alpha=1.0)
    ax4.set_title('Hologram 2 (Y-Z Proj.)', fontsize=10, fontweight='bold')
    ax4.set_xlabel('Y (m)', fontsize=8); ax4.set_ylabel('Z (m)', fontsize=8)
    cbar4 = plt.colorbar(scatter4, ax=ax4, shrink=0.7); cbar4.set_label('Intensity', fontsize=8)
    ax4.set_aspect('equal')

    # 5. Double-exposure Fringes (Y-Z Projection)
    ax5 = fig.add_subplot(3, 3, 5)
    scatter5 = ax5.scatter(y_proj_yz, z_proj_yz, c=double_exposure_intensity_yz, cmap='RdBu', s=0.1, alpha=1.0)
    ax5.set_title('Double-Exposure Fringes\n(Y-Z Projection)', fontsize=10, fontweight='bold', color='red')
    ax5.set_xlabel('Y (m)', fontsize=8); ax5.set_ylabel('Z (m)', fontsize=8)
    cbar5 = plt.colorbar(scatter5, ax=ax5, shrink=0.7); cbar5.set_label('Interference', fontsize=8)
    ax5.set_aspect('equal')

    # 6. Double-exposure Fringes (X-Z Projection - for comparison)
    ax6 = fig.add_subplot(3, 3, 6)
    scatter6 = ax6.scatter(x_proj_xz, z_proj_xz, c=double_exposure_intensity_xz, cmap='RdBu', s=0.1, alpha=1.0)
    ax6.set_title('Double-Exposure Fringes\n(X-Z Projection)', fontsize=10, fontweight='bold', color='blue')
    ax6.set_xlabel('X (m)', fontsize=8); ax6.set_ylabel('Z (m)', fontsize=8)
    cbar6 = plt.colorbar(scatter6, ax=ax6, shrink=0.7); cbar6.set_label('Interference', fontsize=8)
    ax6.set_aspect('equal')
    
    # 7. Cross-section comparison (wider - spans all 3 columns of the bottom row)
    # GridSpec for precise control of bottom row spanning
    gs = fig.add_gridspec(3, 3)
    ax7 = fig.add_subplot(gs[2, :]) # Spans all columns in the 3rd row (index 2)

    center_mask_yz = np.abs(z_proj_yz.flatten()) < (z_proj_yz.max() * 0.1) # Take a thin slice near Z=0
    y_slice_yz = y_proj_yz.flatten()[center_mask_yz]
    
    intensity_orig_slice_yz = intensity_orig_yz.flatten()[center_mask_yz]
    intensity_disp_slice_yz = intensity_disp_yz.flatten()[center_mask_yz]
    double_exp_slice_yz = double_exposure_intensity_yz.flatten()[center_mask_yz]
    
    # Sort by y coordinate for proper line plot
    sort_idx_yz = np.argsort(y_slice_yz)
    y_sorted_yz = y_slice_yz[sort_idx_yz]
    orig_sorted_yz = intensity_orig_slice_yz[sort_idx_yz]
    disp_sorted_yz = intensity_disp_slice_yz[sort_idx_yz]
    double_sorted_yz = double_exp_slice_yz[sort_idx_yz]
    
    ax7.plot(y_sorted_yz, orig_sorted_yz, 'b-', linewidth=1.5, alpha=0.7, label='Hologram 1 (Orig)')
    ax7.plot(y_sorted_yz, disp_sorted_yz, 'r-', linewidth=1.5, alpha=0.7, label='Hologram 2 (Disp)')
    ax7.plot(y_sorted_yz, double_sorted_yz, 'k-', linewidth=2, label='Double-Exposure Fringes')
    ax7.set_title('Cross-section of Y-Z Fringes (slice near Z=0)', fontsize=10, fontweight='bold')
    ax7.set_xlabel('Y position (m)', fontsize=9)
    ax7.set_ylabel('Intensity', fontsize=9)
    ax7.legend(fontsize=8, loc='upper right')
    ax7.grid(True, alpha=0.4)
    
    plt.tight_layout(pad=2.0, h_pad=3.0) # Increased h_pad for row spacing
    plt.show()

def create_vibrating_cube_positions(n_positions=10, amplitude=0.02, frequency_factor=1.0):
    """Create multiple positions of a cube undergoing vibration"""
    vibrating_positions = []
    
    for i in range(n_positions):
        # Sinusoidal vibration in Y direction
        phase = 2 * np.pi * i / n_positions * frequency_factor
        y_displacement = amplitude * np.sin(phase)
        
        # Small random variation in other directions
        x_displacement = amplitude * 0.1 * np.cos(phase * 1.7)
        z_displacement = amplitude * 0.1 * np.sin(phase * 0.8)
        
        displacement_vector = [x_displacement, y_displacement, z_displacement]
        rotation_angle = amplitude * 0.5 * np.sin(phase * 0.5)  # Small rotation
        
        displaced_faces = create_displaced_cube_surface(displacement_vector, rotation_angle)
        vibrating_positions.append(displaced_faces)
    
    return vibrating_positions

def plot_real_time_holographic_interferometry():
    """Demonstrate real-time holographic interferometry"""
    fig = plt.figure(figsize=(12, 10.5))
    
    # Create original cube and a current deformed state
    original_faces = create_cube_surface_full()
    current_faces = create_displaced_cube_surface([0.03, 0.04, -0.01], 0.05)
    
    # Calculate holograms using SIDE view (Y-Z projection)
    X_ref, Y_ref, Z_ref, dist_ref, ref_point = simulate_interferometer_measurement_full(original_faces, 'side')
    X_curr, Y_curr, Z_curr, dist_curr, _ = simulate_interferometer_measurement_full(current_faces, 'side')
    
    # Create interference patterns
    intensity_ref, phase_ref, _ = create_interference_from_distances(dist_ref)
    intensity_curr, phase_curr, _ = create_interference_from_distances(dist_curr)
    
    # Real-time interference (reference hologram vs current object)
    phase_difference = phase_curr - phase_ref
    real_time_intensity = 1 + np.cos(phase_difference)
    displacement_field = dist_curr - dist_ref
    
    # Get Y-Z projections for side view
    z_proj, y_proj = get_projection_for_viewing(X_ref, Y_ref, Z_ref, 'side')
    
    fig.suptitle('Real-Time Holographic Interferometry Demonstration', fontsize=14, fontweight='bold', y=0.98)

    # Manually define subplot positions: [left, bottom, width, height]
    ax1_pos = [0.05, 0.70, 0.4, 0.25]  # Top-left
    ax2_pos = [0.55, 0.70, 0.4, 0.25]  # Top-right (3D)
    ax3_pos = [0.05, 0.37, 0.4, 0.25]  # Middle-left (nudged down slightly)
    ax4_pos = [0.55, 0.37, 0.4, 0.25]  # Middle-right (nudged down slightly)
    ax5_pos = [0.1, 0.06, 0.8, 0.22]   # Bottom, wide (nudged down slightly for title space)

    ax1 = fig.add_axes(ax1_pos)
    ax2 = fig.add_axes(ax2_pos, projection='3d')
    ax3 = fig.add_axes(ax3_pos)
    ax4 = fig.add_axes(ax4_pos)
    ax5 = fig.add_axes(ax5_pos)

    # 1. Reference hologram (Y-Z Projection) (Top-Left)
    scatter1 = ax1.scatter(y_proj, z_proj, c=intensity_ref, cmap='gray', s=0.1, alpha=1.0)
    ax1.set_title('Reference Hologram (Stored Y-Z)', fontsize=9, fontweight='bold')
    ax1.set_xlabel('Y (m)', fontsize=8); ax1.set_ylabel('Z (m)', fontsize=8)
    cbar1 = fig.colorbar(scatter1, ax=ax1, shrink=0.8, aspect=10, pad=0.05)
    cbar1.set_label('Intensity', fontsize=8)
    ax1.set_aspect('equal')
    
    # 2. Current object (live) (Top-Right)
    size = 0.5
    vertices = [
        [-size, -size, -size], [size, -size, -size], [size, size, -size], [-size, size, -size],
        [-size, -size, size], [size, -size, size], [size, size, size], [-size, size, size]
    ]
    disp_vertices = []
    for vertex in vertices:
        cos_theta = np.cos(0.05); sin_theta = np.sin(0.05)
        x_rot = vertex[0] * cos_theta - vertex[1] * sin_theta
        y_rot = vertex[0] * sin_theta + vertex[1] * cos_theta
        z_rot = vertex[2]
        disp_vertices.append([x_rot + 0.03, y_rot + 0.04, z_rot - 0.01])
    cube_faces_viz = [
        [disp_vertices[i] for i in [0,1,2,3]], [disp_vertices[i] for i in [4,5,6,7]],
        [disp_vertices[i] for i in [0,1,5,4]], [disp_vertices[i] for i in [2,3,7,6]],
        [disp_vertices[i] for i in [1,2,6,5]], [disp_vertices[i] for i in [0,3,7,4]]
    ]
    cube_viz = Poly3DCollection(cube_faces_viz, alpha=0.7, facecolor='lightcoral', edgecolor='red', linewidth=1.5)
    ax2.add_collection3d(cube_viz)
    ax2.set_title('Current Object (Live/Deformed)', fontsize=9, fontweight='bold')
    ax2.set_xlabel('X', fontsize=8); ax2.set_ylabel('Y', fontsize=8); ax2.set_zlabel('Z', fontsize=8)
    ax2.view_init(elev=0, azim=90) # Changed to Y-Z projection (side view)
    ax2.set_xlim([-1, 1]); ax2.set_ylim([-1, 1]); ax2.set_zlim([-1, 1])
    
    # 3. Real-time interference fringes (Y-Z Projection) (Middle-Left)
    scatter3 = ax3.scatter(y_proj, z_proj, c=real_time_intensity, cmap='RdBu', s=0.1, alpha=1.0)
    ax3.set_title('Real-Time Interference (Y-Z)', fontsize=9, fontweight='bold', color='red')
    ax3.set_xlabel('Y (m)', fontsize=8); ax3.set_ylabel('Z (m)', fontsize=8)
    cbar3 = fig.colorbar(scatter3, ax=ax3, shrink=0.8, aspect=10, pad=0.05)
    cbar3.set_label('Interference', fontsize=8)
    ax3.set_aspect('equal')
    
    # 4. Displacement field (Y-Z Projection) (Middle-Right)
    scatter4 = ax4.scatter(y_proj, z_proj, c=displacement_field, cmap='viridis', s=0.1, alpha=1.0)
    ax4.set_title('Displacement Field (Y-Z)', fontsize=9, fontweight='bold')
    ax4.set_xlabel('Y (m)', fontsize=8); ax4.set_ylabel('Z (m)', fontsize=8)
    cbar4 = fig.colorbar(scatter4, ax=ax4, shrink=0.8, aspect=10, pad=0.05)
    cbar4.set_label('Displacement (m)', fontsize=8)
    ax4.set_aspect('equal')
    
    # 5. Cross-section showing real-time changes (Bottom, wide plot)
    center_mask = np.abs(z_proj.flatten()) < (z_proj.max() * 0.1) # Take a thin slice near Z=0
    y_slice = y_proj.flatten()[center_mask]
    sort_idx = np.argsort(y_slice)
    y_sorted = y_slice[sort_idx]
    
    intensity_ref_slice = intensity_ref.flatten()[center_mask][sort_idx]
    real_time_slice = real_time_intensity.flatten()[center_mask][sort_idx]
    displacement_slice = displacement_field.flatten()[center_mask][sort_idx]
    
    ax5_twin = ax5.twinx()
    line1 = ax5.plot(y_sorted, intensity_ref_slice, 'b-', linewidth=1.5, alpha=0.7, label='Ref. Hologram')
    line2 = ax5.plot(y_sorted, real_time_slice, 'r-', linewidth=2, label='Real-Time Fringes')
    line3 = ax5_twin.plot(y_sorted, displacement_slice, 'g--', linewidth=1.5, label='Displacement')
    
    ax5.set_title('Cross-section: Real-Time Analysis (slice near Z=0)', fontsize=9, fontweight='bold')
    ax5.set_xlabel('Y position (m)', fontsize=8)
    ax5.set_ylabel('Intensity', fontsize=8, color='black')
    ax5_twin.set_ylabel('Displacement (m)', fontsize=8, color='green')
    ax5.tick_params(axis='y', labelcolor='black', labelsize=7)
    ax5_twin.tick_params(axis='y', labelcolor='green', labelsize=7)
    ax5.grid(True, alpha=0.4)
    
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, loc='upper right', fontsize=7)
    
    plt.show()

def plot_time_averaged_holographic_interferometry():
    """Demonstrate time-averaged holographic interferometry for vibration analysis"""
    fig = plt.figure(figsize=(15, 9)) # Adjusted for a 2x3 layout
    
    # Create vibrating cube positions
    vibrating_positions = create_vibrating_cube_positions(20, amplitude=0.03)
    
    # Calculate holograms for all positions and average them
    all_intensities = []
    all_phases = []
    reference_position = None
    
    for i, faces_data in enumerate(vibrating_positions):
        X_vib, Y_vib, Z_vib, dist_vib, ref_point = simulate_interferometer_measurement_full(faces_data, 'front')
        intensity_vib, phase_vib, _ = create_interference_from_distances(dist_vib)
        
        if i == 0:  # Store reference position and projection
            reference_position = (X_vib, Y_vib, Z_vib)
            x_proj, y_proj = get_projection_for_viewing(X_vib, Y_vib, Z_vib, 'front')
        
        all_intensities.append(intensity_vib)
        all_phases.append(phase_vib)
    
    # Time-averaged intensity (this creates the characteristic vibration pattern)
    time_averaged_intensity = np.mean(all_intensities, axis=0)
    
    # Calculate vibration amplitude from phase variations
    phase_array = np.array(all_phases)
    phase_std = np.std(phase_array, axis=0)
    vibration_amplitude = phase_std * (5e-3) / (2 * np.pi)  # Convert phase to displacement
    
    # 1. Single vibration position
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    size = 0.5
    vertices = [
        [-size, -size, -size], [size, -size, -size], [size, size, -size], [-size, size, -size],
        [-size, -size, size], [size, -size, size], [size, size, size], [-size, size, size]
    ]
    cube_faces_viz = [
        [vertices[i] for i in [0,1,2,3]], [vertices[i] for i in [4,5,6,7]],
        [vertices[i] for i in [0,1,5,4]], [vertices[i] for i in [2,3,7,6]],
        [vertices[i] for i in [1,2,6,5]], [vertices[i] for i in [0,3,7,4]]
    ]
    cube_viz = Poly3DCollection(cube_faces_viz, alpha=0.5, facecolor='lightgreen', edgecolor='green', linewidth=1.5)
    ax1.add_collection3d(cube_viz)
    ax1.quiver(0, 0, 0, 0, 0.15, 0, color='red', arrow_length_ratio=0.2, linewidth=2.5)
    ax1.quiver(0, 0, 0, 0, -0.15, 0, color='red', arrow_length_ratio=0.2, linewidth=2.5)
    ax1.text(0, 0.35, 0, 'Vibration', ha='center', fontsize=9, color='red', fontweight='bold')
    ax1.set_title('Vibrating Object', fontsize=10, fontweight='bold')
    ax1.set_xlabel('X', fontsize=8); ax1.set_ylabel('Y', fontsize=8); ax1.set_zlabel('Z', fontsize=8)
    ax1.view_init(elev=0, azim=0) # Changed to Z-Y projection
    ax1.set_xlim([-1, 1]); ax1.set_ylim([-1, 1]); ax1.set_zlim([-1, 1])
    
    # 2. Individual hologram (single position)
    ax2 = fig.add_subplot(2, 3, 2)
    scatter2 = ax2.scatter(x_proj, y_proj, c=all_intensities[0], cmap='gray', s=0.1, alpha=1.0)
    ax2.set_title('Instantaneous Hologram', fontsize=10, fontweight='bold')
    ax2.set_xlabel('X (m)', fontsize=8); ax2.set_ylabel('Y (m)', fontsize=8)
    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.7); cbar2.set_label('Intensity', fontsize=8)
    ax2.set_aspect('equal')
    
    # 3. Time-averaged hologram
    ax3 = fig.add_subplot(2, 3, 3)
    scatter3 = ax3.scatter(x_proj, y_proj, c=time_averaged_intensity, cmap='plasma', s=0.1, alpha=1.0)
    ax3.set_title('Time-Averaged Hologram', fontsize=10, fontweight='bold', color='red')
    ax3.set_xlabel('X (m)', fontsize=8); ax3.set_ylabel('Y (m)', fontsize=8)
    cbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.7); cbar3.set_label('Avg. Intensity', fontsize=8)
    ax3.set_aspect('equal')
    
    # 4. Vibration amplitude map
    ax4 = fig.add_subplot(2, 3, 4)
    scatter4 = ax4.scatter(x_proj, y_proj, c=vibration_amplitude, cmap='hot', s=0.1, alpha=1.0)
    ax4.set_title('Vibration Amplitude Map', fontsize=10, fontweight='bold')
    ax4.set_xlabel('X (m)', fontsize=8); ax4.set_ylabel('Y (m)', fontsize=8)
    cbar4 = plt.colorbar(scatter4, ax=ax4, shrink=0.7); cbar4.set_label('Amplitude (m)', fontsize=8)
    ax4.set_aspect('equal')
    
    # 5. Comparison of multiple positions (overlay)
    ax5 = fig.add_subplot(2, 3, 5)
    for i in range(0, 20, 5):
        alpha_val = 0.2 + 0.1 * i/5
        ax5.scatter(x_proj, y_proj, c=all_intensities[i], cmap='gray', 
                   s=0.05, alpha=alpha_val, vmin=0, vmax=2)
    ax5.set_title('Overlaid Instantaneous\nHolograms', fontsize=10, fontweight='bold')
    ax5.set_xlabel('X (m)', fontsize=8); ax5.set_ylabel('Y (m)', fontsize=8)
    ax5.set_aspect('equal')
    
    # 6. Cross-section showing vibration (bottom-right corner)
    ax6 = fig.add_subplot(2, 3, 6)
    center_mask = np.abs(y_proj.flatten()) < (y_proj.max() * 0.1)
    x_slice = x_proj.flatten()[center_mask]
    sort_idx = np.argsort(x_slice)
    x_sorted = x_slice[sort_idx]
    
    for i in range(0, 20, 4):
        intensity_slice = all_intensities[i].flatten()[center_mask][sort_idx]
        ax6.plot(x_sorted, intensity_slice, alpha=(0.2 + 0.5 * i/20), linewidth=1, color='blue')
    
    avg_slice = time_averaged_intensity.flatten()[center_mask][sort_idx]
    ax6.plot(x_sorted, avg_slice, 'r-', linewidth=2, label='Time-Averaged')
    
    ax6.set_title('Cross-section (Y ≈ 0)', fontsize=10, fontweight='bold')
    ax6.set_xlabel('X position (m)', fontsize=8)
    ax6.set_ylabel('Intensity', fontsize=8)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.4)
    
    plt.tight_layout(pad=2.0, h_pad=2.5, w_pad=2.5) # Adjusted padding
    plt.show()

if __name__ == "__main__":
    print("Generating high-resolution cube holographic recording visualizations...")
    
    # Create cube visualizations (no automatic saving)
    plot_cube_interferometer_holography()
    plot_cube_holographic_recording()
    plot_holographic_reconstruction()
    plot_double_exposure_holographic_interferometry()
    plot_real_time_holographic_interferometry()
    plot_time_averaged_holographic_interferometry()
    
    print("High-resolution cube visualizations generated (not saved automatically)") 