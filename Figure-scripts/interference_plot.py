import numpy as np
import matplotlib.pyplot as plt

# Set up the figure with modern styling
plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.patch.set_facecolor('white')

# Parameters
x = np.linspace(0, 4*np.pi, 1000)
amplitude = 1.0
frequency = 1.0

# Wave 1 (same for both cases)
wave1 = amplitude * np.sin(frequency * x)

# Wave 2 for constructive interference (in phase)
wave2_constructive = amplitude * np.sin(frequency * x)

# Wave 2 for destructive interference (180° out of phase)
wave2_destructive = amplitude * np.sin(frequency * x + np.pi)

# Calculate superposition
superposition_constructive = wave1 + wave2_constructive
superposition_destructive = wave1 + wave2_destructive

# Colors
color1 = '#2E86AB'  # Blue
color2 = '#A23B72'  # Purple
color_sum = '#F18F01'  # Orange

# Plot constructive interference
ax1.plot(x, wave1, color=color1, linewidth=2, alpha=0.7, label='Wave 1')
ax1.plot(x, wave2_constructive, color=color2, linewidth=2, alpha=0.7, label='Wave 2')
ax1.plot(x, superposition_constructive, color=color_sum, linewidth=3, label='Superposition')
ax1.set_title('Constructive Interference', fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Position', fontsize=12)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)
ax1.set_ylim(-2.5, 2.5)

# Add phase difference annotation
ax1.text(0.5, 2.2, 'Δφ = 0°', fontsize=12, fontweight='bold', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))

# Plot destructive interference
ax2.plot(x, wave1, color=color1, linewidth=2, alpha=0.7, label='Wave 1')
ax2.plot(x, wave2_destructive, color=color2, linewidth=2, alpha=0.7, label='Wave 2')
ax2.plot(x, superposition_destructive, color=color_sum, linewidth=3, label='Superposition')
ax2.set_title('Destructive Interference', fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('Position', fontsize=12)
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)
ax2.set_ylim(-2.5, 2.5)

# Add phase difference annotation
ax2.text(0.5, 2.2, 'Δφ = 180°', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))

# Remove x-axis ticks for cleaner look
for ax in [ax1, ax2]:
    ax.set_xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi])
    ax.set_xticklabels(['0', 'π', '2π', '3π', '4π'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Adjust layout and save
plt.tight_layout()
plt.savefig('wave_interference.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

print("Plot saved as 'wave_interference.png'") 