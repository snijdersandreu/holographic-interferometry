import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set up the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig.patch.set_facecolor('white')

# Colors
beam_color = '#2E86AB'  # Blue for beams
mirror_color = '#666666'  # Gray for mirrors
bs_color = '#A23B72'  # Purple for beam splitters
detector_color = '#F18F01'  # Orange for detectors

# === LEFT PLOT: MICHELSON INTERFEROMETER ===
ax1.set_xlim(-1, 4)
ax1.set_ylim(-1, 4)
ax1.set_aspect('equal')

# Beam splitter (center) - square with diagonal line
bs1_square = patches.Rectangle((1.4, 1.4), 0.2, 0.2, 
                              facecolor='lightgray', edgecolor='black', linewidth=2)
ax1.add_patch(bs1_square)
# Diagonal line inside beam splitter
ax1.plot([1.4, 1.6], [1.4, 1.6], color=bs_color, linewidth=3)

# Mirrors
# Top mirror
mirror1 = patches.Rectangle((1.45, 3.2), 0.1, 0.05, 
                           facecolor=mirror_color, edgecolor='black', linewidth=2)
ax1.add_patch(mirror1)

# Right mirror
mirror2 = patches.Rectangle((3.2, 1.45), 0.05, 0.1, 
                           facecolor=mirror_color, edgecolor='black', linewidth=2)
ax1.add_patch(mirror2)

# Detector
detector1 = patches.Circle((0.5, 1.5), 0.15, 
                          facecolor=detector_color, edgecolor='black', linewidth=2)
ax1.add_patch(detector1)

# Light source
source1 = patches.Circle((1.5, 0.5), 0.15, 
                        facecolor='lightgreen', edgecolor='black', linewidth=2)
ax1.add_patch(source1)

# Beam paths
# Input beam
ax1.arrow(1.5, 0.7, 0, 0.6, head_width=0.08, head_length=0.08, 
          fc=beam_color, ec=beam_color, linewidth=3)

# To top mirror
ax1.arrow(1.5, 1.65, 0, 1.4, head_width=0.08, head_length=0.08, 
          fc=beam_color, ec=beam_color, linewidth=3)

# From top mirror (return)
ax1.arrow(1.5, 3.15, 0, -1.35, head_width=0.08, head_length=0.08, 
          fc=beam_color, ec=beam_color, linewidth=3, alpha=0.7)

# To right mirror
ax1.arrow(1.65, 1.5, 1.4, 0, head_width=0.08, head_length=0.08, 
          fc=beam_color, ec=beam_color, linewidth=3)

# From right mirror (return)
ax1.arrow(3.15, 1.5, -1.35, 0, head_width=0.08, head_length=0.08, 
          fc=beam_color, ec=beam_color, linewidth=3, alpha=0.7)

# To detector
ax1.arrow(1.35, 1.5, -0.7, 0, head_width=0.08, head_length=0.08, 
          fc=beam_color, ec=beam_color, linewidth=3)

# Labels
ax1.text(1.5, 0.2, 'Source', ha='center', fontsize=10, fontweight='bold')
ax1.text(0.5, 1.8, 'Detector', ha='center', fontsize=10, fontweight='bold')
ax1.text(1.5, 3.4, 'Mirror 1', ha='center', fontsize=10, fontweight='bold')
ax1.text(3.4, 1.8, 'Mirror 2', ha='center', fontsize=10, fontweight='bold')
ax1.text(1.8, 1.0, 'Beam\nSplitter', ha='center', fontsize=9, fontweight='bold')

ax1.set_title('Michelson Interferometer', fontsize=16, fontweight='bold', pad=20)
ax1.axis('off')

# === RIGHT PLOT: MACH-ZEHNDER INTERFEROMETER ===
ax2.set_xlim(-1, 5)
ax2.set_ylim(-1, 4)
ax2.set_aspect('equal')

# Beam splitters - squares with diagonal lines
bs2_1_square = patches.Rectangle((0.9, 1.4), 0.2, 0.2, 
                                facecolor='lightgray', edgecolor='black', linewidth=2)
ax2.add_patch(bs2_1_square)
# Diagonal line inside first beam splitter
ax2.plot([0.9, 1.1], [1.4, 1.6], color=bs_color, linewidth=3)

bs2_2_square = patches.Rectangle((3.4, 1.4), 0.2, 0.2, 
                                facecolor='lightgray', edgecolor='black', linewidth=2)
ax2.add_patch(bs2_2_square)
# Diagonal line inside second beam splitter
ax2.plot([3.4, 3.6], [1.4, 1.6], color=bs_color, linewidth=3)

# Mirrors
mirror3 = patches.Rectangle((0.95, 3.2), 0.1, 0.05, 
                           facecolor=mirror_color, edgecolor='black', linewidth=2)
ax2.add_patch(mirror3)

mirror4 = patches.Rectangle((3.45, 3.2), 0.1, 0.05, 
                           facecolor=mirror_color, edgecolor='black', linewidth=2)
ax2.add_patch(mirror4)

# Detectors
detector2 = patches.Circle((4.5, 1.5), 0.15, 
                          facecolor=detector_color, edgecolor='black', linewidth=2)
ax2.add_patch(detector2)

detector3 = patches.Circle((3.5, 0.5), 0.15, 
                          facecolor=detector_color, edgecolor='black', linewidth=2)
ax2.add_patch(detector3)

# Light source
source2 = patches.Circle((0.0, 1.5), 0.15, 
                        facecolor='lightgreen', edgecolor='black', linewidth=2)
ax2.add_patch(source2)

# Beam paths
# Input beam
ax2.arrow(0.2, 1.5, 0.6, 0, head_width=0.08, head_length=0.08, 
          fc=beam_color, ec=beam_color, linewidth=3)

# Upper path
ax2.arrow(1.0, 1.65, 0, 1.4, head_width=0.08, head_length=0.08, 
          fc=beam_color, ec=beam_color, linewidth=3)
ax2.arrow(1.1, 3.25, 2.2, 0, head_width=0.08, head_length=0.08, 
          fc=beam_color, ec=beam_color, linewidth=3)
ax2.arrow(3.5, 3.15, 0, -1.35, head_width=0.08, head_length=0.08, 
          fc=beam_color, ec=beam_color, linewidth=3)

# Lower path
ax2.arrow(1.15, 1.5, 2.15, 0, head_width=0.08, head_length=0.08, 
          fc=beam_color, ec=beam_color, linewidth=3, alpha=0.7)

# Output beams
ax2.arrow(3.65, 1.5, 0.7, 0, head_width=0.08, head_length=0.08, 
          fc=beam_color, ec=beam_color, linewidth=3)
ax2.arrow(3.5, 1.35, 0, -0.7, head_width=0.08, head_length=0.08, 
          fc=beam_color, ec=beam_color, linewidth=3, alpha=0.7)

# Labels
ax2.text(0.0, 1.8, 'Source', ha='center', fontsize=10, fontweight='bold')
ax2.text(4.5, 1.8, 'Detector 1', ha='center', fontsize=10, fontweight='bold')
ax2.text(3.5, 0.2, 'Detector 2', ha='center', fontsize=10, fontweight='bold')
ax2.text(1.0, 3.4, 'Mirror 1', ha='center', fontsize=10, fontweight='bold')
ax2.text(3.5, 3.4, 'Mirror 2', ha='center', fontsize=10, fontweight='bold')
ax2.text(1.0, 1.0, 'BS1', ha='center', fontsize=9, fontweight='bold')
ax2.text(3.5, 1.0, 'BS2', ha='center', fontsize=9, fontweight='bold')

ax2.set_title('Mach-Zehnder Interferometer', fontsize=16, fontweight='bold', pad=20)
ax2.axis('off')

# Adjust layout and save
plt.tight_layout()
plt.savefig('../Figures/interferometer_schemes.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

print("Plot saved as '../Figures/interferometer_schemes.png'") 