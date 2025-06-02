import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def create_cube_surface_full():
    """Create a complete 3D cube surface with ultra-high resolution"""
    size = 1.0
    # Ultra-high resolution for extremely fine patterns
    n_points = 300  # Much higher resolution: 300x300 = 90,000 points per face!
    
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

if __name__ == "__main__":
    print("Generating high-resolution cube holographic recording visualizations...")
    
    # Create cube visualizations (no automatic saving)
    plot_cube_interferometer_holography()
    plot_cube_holographic_recording()
    plot_holographic_reconstruction()
    
    print("High-resolution cube visualizations generated (not saved automatically)") 