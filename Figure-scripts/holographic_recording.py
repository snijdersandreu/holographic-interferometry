import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_object_surface():
    """Create a simple 3D object (a sphere with some surface variation) - ultra high resolution"""
    theta = np.linspace(0, 2*np.pi, 300)  # Same as cube: 300x300
    phi = np.linspace(0, np.pi, 300)      # Same as cube: 300x300  
    THETA, PHI = np.meshgrid(theta, phi)
    
    # Sphere with some surface variation to make it interesting
    R = 1 + 0.15 * np.sin(4*THETA) * np.sin(3*PHI)
    
    X = R * np.sin(PHI) * np.cos(THETA)
    Y = R * np.sin(PHI) * np.sin(THETA)
    Z = R * np.cos(PHI)
    
    return X, Y, Z, R

def simulate_interferometer_measurement(X, Y, Z, measurement_direction='front'):
    """Simulate how an interferometer would measure distances from different sides"""
    if measurement_direction == 'front':
        # Measuring from front (positive Z direction)
        reference_point = np.array([0, 0, 3])
        distances = np.sqrt((X - reference_point[0])**2 + 
                          (Y - reference_point[1])**2 + 
                          (Z - reference_point[2])**2)
        view_mask = Z > 0  # Only see front side
        
    elif measurement_direction == 'side':
        # Measuring from side (positive X direction)
        reference_point = np.array([3, 0, 0])
        distances = np.sqrt((X - reference_point[0])**2 + 
                          (Y - reference_point[1])**2 + 
                          (Z - reference_point[2])**2)
        view_mask = X > 0  # Only see side
        
    elif measurement_direction == 'top':
        # Measuring from top (positive Y direction)
        reference_point = np.array([0, 3, 0])
        distances = np.sqrt((X - reference_point[0])**2 + 
                          (Y - reference_point[1])**2 + 
                          (Z - reference_point[2])**2)
        view_mask = Y > 0  # Only see top side
    
    # Apply visibility mask
    distances_masked = np.where(view_mask, distances, np.nan)
    
    return distances_masked, reference_point

def create_interference_from_distances(distances, wavelength=5e-3, reference_distance=3.0):
    """Create interference pattern from distance measurements"""
    # Calculate path difference
    path_diff = distances - reference_distance
    
    # Phase difference
    phase = 2 * np.pi * path_diff / wavelength
    
    # Interference intensity
    intensity = 1 + np.cos(phase)
    
    return intensity, phase, path_diff

def plot_interferometer_holography():
    """Show how interferometer measurements from different angles create holographic information"""
    fig = plt.figure(figsize=(20, 12))
    
    # Create object
    X, Y, Z, R = create_object_surface()
    
    # 1. 3D Object with measurement directions (large subplot)
    ax1 = fig.add_subplot(2, 4, (1, 2))
    ax1 = fig.add_axes([0.05, 0.55, 0.4, 0.4], projection='3d')
    
    # Plot the object
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, linewidth=0)
    
    # Add measurement directions with arrows
    arrow_length = 1.5
    arrow_props = dict(arrowstyle='->', lw=3, color='red')
    
    # Front measurement
    ax1.quiver(0, 0, 3, 0, 0, -arrow_length, color='red', arrow_length_ratio=0.1, linewidth=3)
    ax1.text(0, 0, 3.5, 'Front\nMeasurement', ha='center', fontsize=10, color='red', fontweight='bold')
    
    # Side measurement  
    ax1.quiver(3, 0, 0, -arrow_length, 0, 0, color='blue', arrow_length_ratio=0.1, linewidth=3)
    ax1.text(3.5, 0, 0, 'Side\nMeasurement', ha='center', fontsize=10, color='blue', fontweight='bold')
    
    # Top measurement
    ax1.quiver(0, 3, 0, 0, -arrow_length, 0, color='green', arrow_length_ratio=0.1, linewidth=3)
    ax1.text(0, 3.5, 0, 'Top\nMeasurement', ha='center', fontsize=10, color='green', fontweight='bold')
    
    ax1.set_title('3D Object with Interferometer\nMeasurement Directions', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.view_init(elev=20, azim=45)
    
    # Remove grid for cleaner look
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
        
        distances, ref_point = simulate_interferometer_measurement(X, Y, Z, direction)
        
        # Create 2D projection for visualization
        if direction == 'front':
            x_proj, y_proj = X, Y
        elif direction == 'side':
            x_proj, y_proj = Z, Y  
        elif direction == 'top':
            x_proj, y_proj = X, Z
            
        # Plot distance map using scatter for ultra-fine detail
        scatter = ax.scatter(x_proj.flatten(), y_proj.flatten(), c=distances.flatten(), 
                           cmap=cmap, s=0.1, alpha=0.9)
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
        
        distances, ref_point = simulate_interferometer_measurement(X, Y, Z, direction)
        intensity, phase, path_diff = create_interference_from_distances(distances)
        
        # Create 2D projection
        if direction == 'front':
            x_proj, y_proj = X, Y
        elif direction == 'side':
            x_proj, y_proj = Z, Y  
        elif direction == 'top':
            x_proj, y_proj = X, Z
            
        # Plot interference pattern using scatter for ultra-fine detail
        scatter = ax.scatter(x_proj.flatten(), y_proj.flatten(), c=intensity.flatten(), 
                           cmap='gray', s=0.1, alpha=0.9)
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

def plot_holographic_recording():
    """Create the main holographic recording visualization - all in one plot"""
    # Create figure with specific size and layout
    fig = plt.figure(figsize=(18, 12))
    
    # Create object surface
    X, Y, Z, R = create_object_surface()
    
    # Calculate distances from front view (most common holographic setup)
    distances, ref_point = simulate_interferometer_measurement(X, Y, Z, 'front')
    intensity, phase, path_diff = create_interference_from_distances(distances)
    
    # 1. 3D Object (top left)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9, linewidth=0, antialiased=True)
    
    # Add interferometer setup
    ax1.quiver(0, 0, 3, 0, 0, -1, color='red', arrow_length_ratio=0.1, linewidth=3)
    ax1.text(0, 0, 3.5, 'Interferometer', ha='center', fontsize=10, color='red', fontweight='bold')
    
    ax1.set_title('3D Object Surface\nwith Interferometer Setup', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('X (m)', fontsize=10)
    ax1.set_ylabel('Y (m)', fontsize=10)
    ax1.set_zlabel('Z (m)', fontsize=10)
    ax1.view_init(elev=25, azim=45)
    # Remove grid for cleaner look
    ax1.grid(False)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    
    # 2. Distance map (top center)
    ax2 = fig.add_subplot(2, 3, 2)
    scatter2 = ax2.scatter(X.flatten(), Y.flatten(), c=distances.flatten(), 
                          cmap='plasma', s=0.1, alpha=0.9)
    ax2.set_title('Distance Measurements\n(Interferometer detects each point)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (m)', fontsize=10)
    ax2.set_ylabel('Y (m)', fontsize=10)
    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
    cbar2.set_label('Distance (m)', fontsize=10)
    ax2.set_aspect('equal')
    
    # 3. Phase map (top right)
    ax3 = fig.add_subplot(2, 3, 3)
    scatter3 = ax3.scatter(X.flatten(), Y.flatten(), c=phase.flatten(), 
                          cmap='hsv', s=0.1, alpha=0.9)
    ax3.set_title('Phase Information\n(Encoded in interference)', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('X (m)', fontsize=10)
    ax3.set_ylabel('Y (m)', fontsize=10)
    cbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.8)
    cbar3.set_label('Phase (radians)', fontsize=10)
    ax3.set_aspect('equal')
    
    # 4. Interference pattern (bottom left)
    ax4 = fig.add_subplot(2, 3, 4)
    scatter4 = ax4.scatter(X.flatten(), Y.flatten(), c=intensity.flatten(), 
                          cmap='gray', s=0.1, alpha=1.0)
    ax4.set_title('Recorded Hologram\n(Interference Pattern)', 
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('X (m)', fontsize=10)
    ax4.set_ylabel('Y (m)', fontsize=10)
    cbar4 = plt.colorbar(scatter4, ax=ax4, shrink=0.8)
    cbar4.set_label('Intensity', fontsize=10)
    ax4.set_aspect('equal')
    
    # 5. Cross-section showing fringes (bottom center)
    ax5 = fig.add_subplot(2, 3, 5)
    
    # Take a slice through the middle for cross-section (same as cube script)
    x_flat = X.flatten()
    y_flat = Y.flatten()
    intensity_flat = intensity.flatten()
    distance_flat = distances.flatten()
    
    center_mask = np.abs(y_flat) < 0.05  # Points near y=0
    x_slice = x_flat[center_mask]
    intensity_slice = intensity_flat[center_mask]
    distance_slice = distance_flat[center_mask]
    
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
        '   to each point on sphere surface',
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
    
    # Adjust layout to prevent overlap
    plt.tight_layout(pad=3.0)
    
    plt.show()

def plot_distance_fringe_relationship():
    """Show how different distances create different fringe patterns"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    wavelength = 5e-3 
    x = np.linspace(-2, 2, 1000)
    
    distances = [2.0, 2.1, 2.2]
    colors = ['blue', 'red', 'green']
    
    for i, (distance, color) in enumerate(zip(distances, colors)):
        # Simple model: path difference increases with distance from center
        path_diff_1d = np.abs(x) + distance - 2.0  # Reference distance = 2.0
        phase_1d = 2 * np.pi * path_diff_1d / wavelength
        intensity_1d = 1 + np.cos(phase_1d)
        
        axes[i].plot(x, intensity_1d, color=color, linewidth=2, 
                    label=f'Distance: {distance}m')
        axes[i].set_title(f'Object at distance {distance}m', fontweight='bold')
        axes[i].set_xlabel('Position (m)')
        axes[i].set_ylabel('Intensity')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, 2)
        axes[i].legend()
    
    fig.suptitle('How Different Object Distances Create Different Fringe Patterns', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_hologram_formation_concept():
    """Create a conceptual diagram showing hologram formation"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create a simple 2D representation
    x = np.linspace(-3, 3, 300)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    
    # Simulate object points at different distances
    object_points = [(-1, 0, 1), (0, 0, 1.5), (1, 0, 2)]
    wavelength = 1.0  # Normalized wavelength
    
    # Recording plane at z = 0
    total_intensity = np.ones_like(X)
    
    for i, (ox, oy, oz) in enumerate(object_points):
        # Distance from object point to each point on recording plane
        distance = np.sqrt((X - ox)**2 + (Y - oy)**2 + oz**2)
        
        # Reference beam distance (constant)
        ref_distance = 3.0
        
        # Phase difference
        phase_diff = 2 * np.pi * (distance - ref_distance) / wavelength
        
        # Add interference pattern
        total_intensity += 0.5 * (1 + np.cos(phase_diff))
    
    # Plot the interference pattern
    im = ax.contourf(X, Y, total_intensity, levels=50, cmap='gray')
    
    # Add object points
    for i, (ox, oy, oz) in enumerate(object_points):
        ax.plot(ox, oy, 'ro', markersize=10, label=f'Object point {i+1} (z={oz})')
    
    ax.set_title('Hologram Formation: Multiple Object Points\nCreate Complex Interference Pattern', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.legend()
    
    # Add text explanation
    ax.text(0.02, 0.98, 'Each object point creates\ncircular interference fringes.\nAll patterns superimpose\nto form the hologram.', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.colorbar(im, ax=ax, label='Intensity')
    plt.tight_layout()
    plt.show()

def plot_holographic_reconstruction():
    """Demonstrate the holographic reconstruction process for sphere"""
    fig = plt.figure(figsize=(16, 10))
    
    # Create sphere and get hologram data
    X, Y, Z, R = create_object_surface()
    distances, ref_point = simulate_interferometer_measurement(X, Y, Z, 'front')
    intensity, phase, path_diff = create_interference_from_distances(distances)
    
    # 1. Original 3D Object (top left)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
    ax1.set_title('1. Original Object\n(Complex Surface)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.view_init(elev=25, azim=45)
    ax1.set_xlim([-2, 2]); ax1.set_ylim([-2, 2]); ax1.set_zlim([-2, 2])
    ax1.grid(False)
    
    # 2. Recorded Hologram (top center)
    ax2 = fig.add_subplot(2, 3, 2)
    scatter2 = ax2.scatter(X.flatten(), Y.flatten(), c=intensity.flatten(), 
                          cmap='gray', s=0.1, alpha=1.0)
    ax2.set_title('2. Recorded Hologram\n(Interference Pattern)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
    plt.colorbar(scatter2, ax=ax2, shrink=0.7, label='Intensity')
    ax2.set_aspect('equal')
    
    # 3. Reconstruction Beam Setup (top right)
    ax3 = fig.add_subplot(2, 3, 3)
    # Show hologram plane and reconstruction beam
    hologram_x = np.linspace(-1.5, 1.5, 100)
    hologram_y = np.linspace(-1.5, 1.5, 100)
    H_x, H_y = np.meshgrid(hologram_x, hologram_y)
    
    # Reconstruction beam (parallel wavefront)
    ax3.contourf(H_x, H_y, np.ones_like(H_x), levels=1, colors=['lightcyan'], alpha=0.7)
    ax3.contour(H_x, H_y, np.ones_like(H_x), levels=1, colors=['blue'], linewidths=2)
    
    # Add arrows showing beam direction
    for i in range(0, len(hologram_x), 15):
        ax3.arrow(hologram_x[i], 1.8, 0, -0.2, head_width=0.05, head_length=0.03, fc='blue', ec='blue')
    
    ax3.text(0, 2.0, 'Reconstruction Beam', ha='center', fontsize=10, color='blue', fontweight='bold')
    ax3.set_title('3. Reconstruction Setup\n(Beam illuminates hologram)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X (m)'); ax3.set_ylabel('Y (m)')
    ax3.set_xlim(-1.7, 1.7); ax3.set_ylim(-1.7, 2.2)
    ax3.set_aspect('equal')
    
    # 4. Diffraction Pattern (bottom left)
    ax4 = fig.add_subplot(2, 3, 4)
    # Simulate diffraction by showing how fringes scatter light
    diffraction_angles = np.linspace(-np.pi/3, np.pi/3, 60)
    fringe_spacing = np.linspace(0.005, 0.15, 60)  # Different fringe spacings
    ANGLES, SPACING = np.meshgrid(diffraction_angles, fringe_spacing)
    
    # Diffraction efficiency (more efficient for finer fringes)
    diffraction_efficiency = np.exp(-SPACING * 30) * np.cos(ANGLES)**2
    
    scatter4 = ax4.contourf(ANGLES * 180/np.pi, SPACING * 1000, diffraction_efficiency, 
                           levels=25, cmap='plasma')
    ax4.set_title('4. Diffraction Pattern\n(Hologram acts as grating)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Diffraction Angle (degrees)')
    ax4.set_ylabel('Fringe Spacing (mm)')
    plt.colorbar(scatter4, ax=ax4, shrink=0.7, label='Efficiency')
    
    # 5. Reconstructed Wavefront (bottom center)
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    
    # Create a "reconstructed" sphere (slightly transparent to show it's virtual)
    surf_reconstructed = ax5.plot_surface(X, Y, Z, cmap='Reds', alpha=0.4, linewidth=0, 
                                        antialiased=True, linestyle='--')
    
    # Add some wavefront lines to show reconstruction
    for radius in [0.5, 1.0, 1.5]:
        theta_wave = np.linspace(0, 2*np.pi, 30)
        phi_wave = np.linspace(0, np.pi, 15)
        THETA_w, PHI_w = np.meshgrid(theta_wave, phi_wave)
        X_wave = radius * np.sin(PHI_w) * np.cos(THETA_w)
        Y_wave = radius * np.sin(PHI_w) * np.sin(THETA_w)
        Z_wave = radius * np.cos(PHI_w)
        ax5.plot_wireframe(X_wave, Y_wave, Z_wave, alpha=0.3, color='red', linewidth=0.5)
    
    ax5.set_title('5. Reconstructed Object\n(Virtual 3D image)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('X'); ax5.set_ylabel('Y'); ax5.set_zlabel('Z')
    ax5.view_init(elev=25, azim=45)
    ax5.set_xlim([-2, 2]); ax5.set_ylim([-2, 2]); ax5.set_zlim([-2, 2])
    ax5.grid(False)
    
    # 6. Viewing from Different Angles (bottom right)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.text(0.5, 0.9, 'Holographic Reconstruction', ha='center', va='center', 
             fontsize=14, fontweight='bold')
    
    reconstruction_steps = [
        '• Illuminate hologram with coherent light',
        '• Complex interference fringes act as diffraction grating',
        '• Light diffracts at multiple specific angles',
        '• Diffracted beams reconstruct original wavefront',
        '• Observer sees 3D virtual image of complex surface',
        '• Full parallax - move around to see all sides!'
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
    print("Generating ultra-high resolution holographic recording visualizations...")
    
    # Create all visualizations (no automatic saving)
    plot_interferometer_holography()  # Multi-angle measurements
    plot_holographic_recording()      # Main recording process
    plot_distance_fringe_relationship()
    plot_hologram_formation_concept()
    plot_holographic_reconstruction()
    
    print("Ultra-high resolution visualizations generated (not saved automatically)") 