import numpy as np
import matplotlib.pyplot as plt

# Set up the figure with modern styling
plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.patch.set_facecolor('white')

# Colors (same as previous script)
color1 = '#2E86AB'  # Blue
color2 = '#A23B72'  # Purple
color3 = '#F18F01'  # Orange
color4 = '#C73E1D'  # Red

# === LEFT PLOT: TEMPORAL COHERENCE ===
x_time = np.linspace(0, 10, 1000)

# High temporal coherence (narrow bandwidth, long coherence time)
freq_narrow = 1.0
bandwidth_narrow = 0.05
envelope_narrow = np.exp(-0.5 * ((x_time - 5) / 2)**2)
wave_narrow = envelope_narrow * np.cos(2 * np.pi * freq_narrow * x_time)

# Low temporal coherence (broad bandwidth, short coherence time)
freq_broad = 1.0
bandwidth_broad = 0.3
envelope_broad = np.exp(-0.5 * ((x_time - 5) / 0.8)**2)
wave_broad = envelope_broad * np.cos(2 * np.pi * freq_broad * x_time)

ax1.plot(x_time, wave_narrow, color=color1, linewidth=2.5, label='High Temporal Coherence\n(Narrow bandwidth)')
ax1.plot(x_time, wave_broad + 2.5, color=color2, linewidth=2.5, label='Low Temporal Coherence\n(Broad bandwidth)')

ax1.set_title('Temporal Coherence', fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Time', fontsize=12)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10, loc='upper right')
ax1.set_ylim(-1.5, 4)

# Add coherence length annotations
ax1.annotate('Long coherence time', xy=(5, 0.8), xytext=(7, 1.5),
            arrowprops=dict(arrowstyle='->', color=color1, lw=1.5),
            fontsize=10, color=color1, fontweight='bold')
ax1.annotate('Short coherence time', xy=(5, 3.3), xytext=(2.5, 4.5),
            arrowprops=dict(arrowstyle='->', color=color2, lw=1.5),
            fontsize=10, color=color2, fontweight='bold')

# === RIGHT PLOT: SPATIAL COHERENCE ===
# Create 2D spatial coherence visualization
x_space = np.linspace(-3, 3, 100)
y_space = np.linspace(-2, 2, 80)
X, Y = np.meshgrid(x_space, y_space)

# High spatial coherence (small source, uniform phase)
coherent_source = np.exp(-(X**2 + Y**2) / 0.3)
phase_coherent = np.zeros_like(X)
wave_coherent = coherent_source * np.cos(2 * np.pi * 2 * X + phase_coherent)

# Low spatial coherence (large source, varying phase)
incoherent_source = np.exp(-(X**2 + Y**2) / 2.0)
phase_incoherent = 3 * np.random.random(X.shape) * np.pi
wave_incoherent = incoherent_source * np.cos(2 * np.pi * 2 * X + phase_incoherent)

# Plot spatial coherence as 1D cross-sections for clarity
y_slice = 40  # middle row
ax2.plot(x_space, wave_coherent[y_slice, :], color=color3, linewidth=2.5, 
         label='High Spatial Coherence\n(Small/collimated source)')
ax2.plot(x_space, wave_incoherent[y_slice, :] - 2, color=color4, linewidth=2.5, 
         label='Low Spatial Coherence\n(Large/extended source)')

ax2.set_title('Spatial Coherence', fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('Spatial Position', fontsize=12)
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10, loc='upper left')
ax2.set_ylim(-3, 1.5)

# Add spatial coherence annotations
ax2.annotate('Uniform phase\nacross wavefront', xy=(0, 0.8), xytext=(1.5, 1.2),
            arrowprops=dict(arrowstyle='->', color=color3, lw=1.5),
            fontsize=10, color=color3, fontweight='bold', ha='center')
ax2.annotate('Random phase\nvariations', xy=(0, -2.8), xytext=(-1.5, -0.5),
            arrowprops=dict(arrowstyle='->', color=color4, lw=1.5),
            fontsize=10, color=color4, fontweight='bold', ha='center')

# Clean up axes
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])  # Remove x-axis ticks for cleaner look
    ax.set_yticks([])  # Remove y-axis ticks for cleaner look

# Adjust layout and save
plt.tight_layout()
plt.savefig('../Figures/coherence_comparison.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

print("Plot saved as '../Figures/coherence_comparison.png'") 