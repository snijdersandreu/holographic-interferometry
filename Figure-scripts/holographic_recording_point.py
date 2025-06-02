import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_recording_plane(size=2.0, resolution=400):
    """Create a high-resolution recording plane (hologram detector)"""
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)  # Recording plane at z=0
    return X, Y, Z

def simulate_point_hologram(point_position, recording_plane, reference_point, wavelength=5e-3):
    """Simulate hologram formation from a single point"""
    X, Y, Z = recording_plane
    px, py, pz = point_position
    rx, ry, rz = reference_point
    
    # Distance from point to each position on recording plane
    object_distances = np.sqrt((X - px)**2 + (Y - py)**2 + (Z - pz)**2)
    
    # Distance from reference point to recording plane (constant for plane wave)
    reference_distance = rz  # For simplicity, assume reference is at (0,0,rz)
    
    # Path difference
    path_difference = object_distances - reference_distance
    
    # Phase difference
    phase = 2 * np.pi * path_difference / wavelength
    
    # Interference intensity (object + reference beam)
    intensity = 1 + np.cos(phase)  # Simplified: assumes equal amplitudes
    
    return intensity, phase, path_difference, object_distances

def plot_single_point_holography():
    """Demonstrate holography of a single point"""
    fig = plt.figure(figsize=(16, 12))
    
    # Parameters
    wavelength = 5e-3
    point_position = (0, 0, 1.0)  # Point at z=1m
    reference_point = (0, 0, 3.0)  # Reference beam from z=3m
    
    # Create recording plane
    recording_plane = create_recording_plane(size=1.5, resolution=500)
    X, Y, Z = recording_plane
    
    # Calculate hologram
    intensity, phase, path_diff, obj_distances = simulate_point_hologram(
        point_position, recording_plane, reference_point, wavelength)
    
    # 1. 3D Setup (top left)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    # Plot recording plane
    ax1.plot_surface(X, Y, Z, alpha=0.3, color='lightblue', label='Recording Plane')
    
    # Plot point object
    px, py, pz = point_position
    ax1.scatter([px], [py], [pz], color='red', s=100, label='Point Object')
    
    # Plot reference beam direction
    ax1.quiver(0, 0, 3, 0, 0, -2, color='blue', arrow_length_ratio=0.1, linewidth=3)
    ax1.text(0, 0, 3.5, 'Reference\nBeam', ha='center', color='blue', fontweight='bold')
    
    # Show some light rays from point to recording plane
    for i in range(0, len(X[0]), 100):
        for j in range(0, len(Y), 100):
            ax1.plot([px, X[j,i]], [py, Y[j,i]], [pz, Z[j,i]], 'r-', alpha=0.2, linewidth=0.5)
    
    ax1.set_title('Single Point Holography Setup', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
    ax1.view_init(elev=25, azim=45)
    ax1.set_xlim([-2, 2]); ax1.set_ylim([-2, 2]); ax1.set_zlim([0, 4])
    
    # 2. Distance Map (top center)
    ax2 = fig.add_subplot(2, 3, 2)
    contour2 = ax2.contourf(X, Y, obj_distances, levels=50, cmap='plasma')
    ax2.set_title('Distance from Point\nto Recording Plane', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
    plt.colorbar(contour2, ax=ax2, shrink=0.8, label='Distance (m)')
    ax2.set_aspect('equal')
    
    # 3. Phase Map (top right)
    ax3 = fig.add_subplot(2, 3, 3)
    contour3 = ax3.contourf(X, Y, phase, levels=50, cmap='hsv')
    ax3.set_title('Phase Difference\n(Path difference × 2π/λ)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X (m)'); ax3.set_ylabel('Y (m)')
    plt.colorbar(contour3, ax=ax3, shrink=0.8, label='Phase (radians)')
    ax3.set_aspect('equal')
    
    # 4. Interference Pattern - Full (bottom left)
    ax4 = fig.add_subplot(2, 3, 4)
    contour4 = ax4.contourf(X, Y, intensity, levels=50, cmap='gray')
    ax4.set_title('Hologram Pattern\n(Circular Fringes)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('X (m)'); ax4.set_ylabel('Y (m)')
    plt.colorbar(contour4, ax=ax4, shrink=0.8, label='Intensity')
    ax4.set_aspect('equal')
    
    # 5. Cross-section (bottom center)
    ax5 = fig.add_subplot(2, 3, 5)
    
    # Take cross-section through center
    center_idx = len(Y) // 2
    x_cross = X[center_idx, :]
    intensity_cross = intensity[center_idx, :]
    distance_cross = obj_distances[center_idx, :]
    
    # Plot both intensity and distance
    ax5_twin = ax5.twinx()
    line1 = ax5.plot(x_cross, intensity_cross, 'b-', linewidth=2, label='Interference intensity')
    line2 = ax5_twin.plot(x_cross, distance_cross, 'r--', linewidth=2, label='Distance to point')
    
    ax5.set_title('Cross-section Through Center', fontsize=12, fontweight='bold')
    ax5.set_xlabel('X position (m)')
    ax5.set_ylabel('Intensity', color='blue')
    ax5_twin.set_ylabel('Distance (m)', color='red')
    ax5.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, loc='upper right')
    
    # 6. Explanation (bottom right)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.text(0.5, 0.9, 'Single Point Holography', ha='center', va='center', 
             fontsize=14, fontweight='bold')
    
    explanations = [
        '• Point source creates spherical waves',
        '• Distance varies across recording plane',
        '• Creates circular interference fringes',
        '• Fringe spacing ∝ 1/distance to point',
        '• Pattern encodes 3D position of point',
        '• Reconstruction reverses the process!'
    ]
    
    for i, text in enumerate(explanations):
        color = 'red' if 'Reconstruction' in text else 'black'
        weight = 'bold' if 'Reconstruction' in text else 'normal'
        ax6.text(0.05, 0.75 - i*0.1, text, ha='left', va='center', 
                fontsize=10, color=color, fontweight=weight)
    
    ax6.set_xlim(0, 1); ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    plt.tight_layout(pad=2.0)
    plt.show()

def plot_multiple_points_holography():
    """Show how multiple points create complex interference patterns"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    wavelength = 5e-3
    reference_point = (0, 0, 3.0)
    recording_plane = create_recording_plane(size=1.5, resolution=300)
    X, Y, Z = recording_plane
    
    # Different point configurations
    point_configs = [
        [(0, 0, 1.0)],  # Single point
        [(0, 0, 1.0), (0.5, 0, 1.2)],  # Two points
        [(0, 0, 1.0), (0.5, 0, 1.2), (-0.3, 0.4, 0.8)]  # Three points
    ]
    
    titles = ['Single Point', 'Two Points', 'Three Points']
    
    for col, (points, title) in enumerate(zip(point_configs, titles)):
        # Calculate combined hologram
        total_intensity = np.ones_like(X)
        
        for point_pos in points:
            intensity, _, _, _ = simulate_point_hologram(
                point_pos, recording_plane, reference_point, wavelength)
            total_intensity += 0.5 * intensity  # Add contributions
        
        # Top row: Setup visualization
        ax_setup = axes[0, col]
        ax_setup.set_title(f'{title}\nSetup', fontsize=12, fontweight='bold')
        
        # Show recording plane as background
        ax_setup.contourf(X, Y, np.ones_like(X), levels=1, colors=['lightblue'], alpha=0.3)
        
        # Show points projected onto recording plane
        for i, point_pos in enumerate(points):
            px, py, pz = point_pos
            ax_setup.plot(px, py, 'ro', markersize=10, label=f'Point {i+1} (z={pz}m)')
            
            # Draw circles showing wavefronts
            radii = np.linspace(0.1, 1.0, 5)
            for r in radii:
                circle = plt.Circle((px, py), r, fill=False, color='red', alpha=0.3, linestyle='--')
                ax_setup.add_patch(circle)
        
        ax_setup.set_xlabel('X (m)'); ax_setup.set_ylabel('Y (m)')
        ax_setup.set_xlim(-1.5, 1.5); ax_setup.set_ylim(-1.5, 1.5)
        ax_setup.set_aspect('equal')
        ax_setup.legend(fontsize=8)
        
        # Bottom row: Resulting hologram
        ax_hologram = axes[1, col]
        contour = ax_hologram.contourf(X, Y, total_intensity, levels=50, cmap='gray')
        ax_hologram.set_title(f'{title}\nHologram Pattern', fontsize=12, fontweight='bold')
        ax_hologram.set_xlabel('X (m)'); ax_hologram.set_ylabel('Y (m)')
        ax_hologram.set_aspect('equal')
        plt.colorbar(contour, ax=ax_hologram, shrink=0.7)
    
    plt.tight_layout()
    plt.show()

def plot_point_distance_effects():
    """Show how point distance affects fringe spacing"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    wavelength = 5e-3
    reference_point = (0, 0, 3.0)
    recording_plane = create_recording_plane(size=1.0, resolution=200)
    X, Y, Z = recording_plane
    
    # Different distances
    distances = [0.5, 1.0, 2.0]
    
    for i, distance in enumerate(distances):
        point_position = (0, 0, distance)
        intensity, _, _, _ = simulate_point_hologram(
            point_position, recording_plane, reference_point, wavelength)
        
        contour = axes[i].contourf(X, Y, intensity, levels=30, cmap='gray')
        axes[i].set_title(f'Point at z = {distance}m\n(Closer → Finer Fringes)', 
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel('X (m)'); axes[i].set_ylabel('Y (m)')
        axes[i].set_aspect('equal')
        plt.colorbar(contour, ax=axes[i], shrink=0.8)
    
    fig.suptitle('Effect of Point Distance on Fringe Spacing', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Generating point holography visualizations...")
    
    # Create all visualizations
    plot_single_point_holography()      # Main single point demo
    plot_multiple_points_holography()   # Show superposition
    plot_point_distance_effects()       # Show distance effects
    
    print("Point holography visualizations generated!") 