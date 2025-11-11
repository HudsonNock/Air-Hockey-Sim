import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load the CSV file

# ============================================================================
# ANALYSIS MODE SELECTION
# ============================================================================
# Set to 'unobstructed' or 'obstructed'
ANALYSIS_MODE = 'obstructed'  # Change this to switch modes
# ============================================================================

if ANALYSIS_MODE == 'unobstructed':
    df = pd.read_csv('puck_data_noise.csv')
else:
    df = pd.read_csv('puck_data_noise_beam.csv')

print(f"\n{'='*90}")
print(f"ANALYSIS MODE: {ANALYSIS_MODE.upper()}")
print(f"{'='*90}\n")

# Parameters
if ANALYSIS_MODE == 'unobstructed':
    window_size = 20  # Number of points for sliding window
    velocity_threshold = 5e-5  # Velocity threshold for stationary (adjust as needed)
    min_velocity_threshold = -1
    min_segment_length = 30  # Minimum points to consider a stationary segment
    trim_points = 10  # Points to trim from each end of segments
else:
    window_size = 20  # Number of points for sliding window
    velocity_threshold = 1e-3  # Velocity threshold for stationary (adjust as needed)
    min_velocity_threshold = 1e-6
    min_segment_length = 30  # Minimum points to consider a stationary segment
    trim_points = 10  # Points to trim from each end of segments

# Calculate velocity using windowed linear regression
def calculate_windowed_velocity(df, window_size):
    """
    Calculate velocity using linear fit over a sliding window.
    Returns velocity magnitude at each point.
    """
    velocities = np.full(len(df), np.nan)
    
    for i in range(window_size // 2, len(df) - window_size // 2):
        # Extract window centered at i
        start_idx = i - window_size // 2
        end_idx = i + window_size // 2
        window = df.iloc[start_idx:end_idx]
        
        # Create time indices for the window
        t = np.arange(len(window))
        
        # Linear fit for x vs t
        slope_x, _, _, _, _ = stats.linregress(t, window['x'].values)
        
        # Linear fit for y vs t
        slope_y, _, _, _, _ = stats.linregress(t, window['y'].values)
        
        # Velocity magnitude (Euclidean norm of slopes)
        velocities[i] = np.sqrt(slope_x**2 + slope_y**2)
    
    return velocities

print("Calculating windowed velocities...")
df['velocity'] = calculate_windowed_velocity(df, window_size)

# Remove NaN values at the edges
valid_df = df.dropna(subset=['velocity']).copy()

# Identify stationary points
valid_df['is_stationary'] = (valid_df['velocity'] < velocity_threshold) & (valid_df['velocity'] > min_velocity_threshold)

# Find continuous stationary segments
valid_df['segment_id'] = (valid_df['is_stationary'] != valid_df['is_stationary'].shift()).cumsum()

# Filter for only stationary segments
stationary_segments = []
for seg_id, group in valid_df[valid_df['is_stationary']].groupby('segment_id'):
    if len(group) >= min_segment_length + 2 * trim_points:
        # Trim endpoints
        trimmed_group = group.iloc[trim_points:-trim_points].copy()
        
        if ANALYSIS_MODE == 'unobstructed':
            # Mode 1: Only use non-obstructed AND visible points
            obstructed_bool = np.array(trimmed_group['obstructed'], dtype=np.bool_)
            visable_bool = np.array(trimmed_group['visable'], dtype=np.bool_)
            filter_mask = (~obstructed_bool) & (visable_bool)
            filtered_data = trimmed_group[filter_mask]
            
            # Track what was filtered out
            n_total = len(trimmed_group)
            n_filtered = len(filtered_data)
            pct_obstructed = (obstructed_bool.sum() / n_total * 100) if n_total > 0 else 0
            pct_not_visible = ((~visable_bool).sum() / n_total * 100) if n_total > 0 else 0
            
        else:  # ANALYSIS_MODE == 'obstructed'
            # Mode 2: Only use obstructed BUT visible points
            obstructed_bool = np.array(trimmed_group['obstructed'], dtype=np.bool_)
            visable_bool = np.array(trimmed_group['visable'], dtype=np.bool_)
            filter_mask = (obstructed_bool) & (visable_bool)
            filtered_data = trimmed_group[filter_mask]
            
            # Calculate percentages relative to obstructed+visible points
            obstructed_and_visible = obstructed_bool & visable_bool
            n_obstructed_visible = obstructed_and_visible.sum()
            
            if n_obstructed_visible > 0:
                # Of the obstructed+visible points, what % are obstructed?
                pct_obstructed = 100.0  # By definition, all filtered points are obstructed
                # Of the obstructed+visible points, what % are not visible?
                # This doesn't make sense in this mode, so we track something else
                pct_not_visible = (((~visable_bool) & (obstructed_bool)).sum() / obstructed_bool.sum() * 100)
            else:
                pct_obstructed = 0
                pct_not_visible = 0
            
            n_total = obstructed_bool.sum()
            n_filtered = len(filtered_data)
        
        # Calculate statistics only on filtered points
        if len(filtered_data) > 0:
            mean_x = filtered_data['x'].mean()
            mean_y = filtered_data['y'].mean()
            std_x = filtered_data['x'].std()
            std_y = filtered_data['y'].std()
            
            # Calculate maximum deviation from mean using filtered points
            deviations = np.sqrt((filtered_data['x'].values - mean_x)**2 + 
                                (filtered_data['y'].values - mean_y)**2)
            max_deviation = np.max(deviations)
        else:
            # Fallback if no points pass the filter
            mean_x = trimmed_group['x'].mean()
            mean_y = trimmed_group['y'].mean()
            std_x = 0.0
            std_y = 0.0
            max_deviation = 0.0
        
        stationary_segments.append({
            'segment_id': seg_id,
            'start_idx': trimmed_group.index[0],
            'end_idx': trimmed_group.index[-1],
            'length': n_total,
            'n_filtered': n_filtered,
            'pct_obstructed': pct_obstructed,
            'pct_not_visible': pct_not_visible,
            'mean_x': mean_x,
            'mean_y': mean_y,
            'std_x': std_x,
            'std_y': std_y,
            'max_deviation': max_deviation,
            'mean_velocity': trimmed_group['velocity'].mean(),
            'data': trimmed_group,
            'filter_mask': filter_mask
        })

print(f"\nFound {len(stationary_segments)} stationary segments")
print("\nSegment Summary:")
print("-" * 90)
total_obstructed = 0
total_not_visable = 0
for i, seg in enumerate(stationary_segments):
    total_obstructed += seg['length']
    total_not_visable += seg['pct_not_visible'] * seg['length']
    print(f"Segment {i+1}:")
    print(f"  Indices: {seg['start_idx']} to {seg['end_idx']} ({seg['length']} points, {seg['n_filtered']} filtered)")
    print(f"  Obstructed: {seg['pct_obstructed']:.1f}%, Not visible: {seg['pct_not_visible']:.1f}%")
    print(f"  Mean position (filtered pts): x={seg['mean_x']:.3f}, y={seg['mean_y']:.3f}")
    print(f"  Mean velocity: {seg['mean_velocity']:.4f}")
    print(f"  Std deviation (filtered pts): x={seg['std_x']:.4f}, y={seg['std_y']:.4f}")
    print(f"  Max deviation from mean (filtered pts): {seg['max_deviation']:.4f}")
    print()

print("Total not visable percentage:")
print(total_not_visable / total_obstructed)

# Visualize the segmentation
plt.figure(figsize=(14, 10))

# Plot 1: Full trajectory with segments highlighted
plt.subplot(2, 2, 1)
plt.plot(df['x'], df['y'], 'gray', alpha=0.3, linewidth=0.5, label='Full trajectory')
colors = plt.cm.rainbow(np.linspace(0, 1, len(stationary_segments)))
for i, seg in enumerate(stationary_segments):
    seg_data = seg['data']
    # Plot all points in light color
    plt.scatter(seg_data['x'], seg_data['y'], c=[colors[i]], s=10, alpha=0.3)
    # Overlay filtered points in solid color
    filtered_data = seg_data[seg['filter_mask']]
    mode_label = 'unobs' if ANALYSIS_MODE == 'unobstructed' else 'obs'
    plt.scatter(filtered_data['x'], filtered_data['y'], c=[colors[i]], s=15, 
                label=f'Seg {i+1} ({mode_label}, {seg["n_filtered"]} pts)')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Trajectory - {ANALYSIS_MODE.capitalize()} Mode (Filtered Points Highlighted)')
#plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Windowed velocity over time
plt.subplot(2, 2, 2)
plt.plot(valid_df.index, valid_df['velocity'], 'b-', alpha=0.5, linewidth=0.5)
plt.axhline(y=velocity_threshold, color='r', linestyle='--', label=f'Threshold = {velocity_threshold}')
for seg in stationary_segments:
    plt.axvspan(seg['start_idx'], seg['end_idx'], alpha=0.2, color='green')
plt.xlabel('Index')
plt.ylabel('Windowed Velocity')
plt.title(f'Windowed Velocity (window={window_size}) - Stationary Segments Highlighted')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Plot 3: Noise and Max Deviation vs distance for each segment
plt.subplot(2, 2, 3)
distances = [seg['mean_x'] for seg in stationary_segments]
noise_x = [seg['std_x'] for seg in stationary_segments]
noise_y = [seg['std_y'] for seg in stationary_segments]
max_devs = [seg['max_deviation'] for seg in stationary_segments]
noise_combined = [np.sqrt(seg['std_x']**2 + seg['std_y']**2) for seg in stationary_segments]
visable = [seg['pct_not_visible'] for seg in stationary_segments]

plt.scatter(distances, noise_x, label='Std X', marker='o', s=80, alpha=0.7)
plt.scatter(distances, noise_y, label='Std Y', marker='s', s=80, alpha=0.7)
plt.scatter(distances, noise_combined, label='Std Combined', marker='^', s=80, alpha=0.7)
plt.scatter(distances, max_devs, label='Max Deviation', marker='D', s=100, alpha=0.7, edgecolors='black', linewidths=1.5)
plt.xlabel('Distance (x position)')
plt.ylabel('Noise / Deviation')
plt.title(f'Sensor Noise vs Distance - {ANALYSIS_MODE.capitalize()} Mode')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Individual segment scatter plots
plt.subplot(2, 2, 4)
for i, seg in enumerate(stationary_segments):
    seg_data = seg['data']
    filtered_data = seg_data[seg['filter_mask']]
    # Plot all points lightly
    plt.scatter(seg_data['x'], seg_data['y'], c=[colors[i]], s=20, alpha=0.2)
    # Overlay filtered points
    plt.scatter(filtered_data['x'], filtered_data['y'], c=[colors[i]], s=30, alpha=0.8, label=f'Seg {i+1}')
    # Plot mean position
    plt.scatter(seg['mean_x'], seg['mean_y'], c=[colors[i]], s=200, marker='x', linewidths=3)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Stationary Segments - {ANALYSIS_MODE.capitalize()} (X = Mean)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Analyze noise vs distance relationship
print("distances")
print(distances)
if len(stationary_segments) > 1:
    print("\nNoise and Max Deviation Analysis:")
    print("-" * 90)
    
    # Fit linear model: noise vs distance
    from scipy.stats import linregress
    
    slope_std_x, intercept_std_x, r_std_x, _, _ = linregress(distances, noise_x)
    slope_std_y, intercept_std_y, r_std_y, _, _ = linregress(distances, noise_y)
    slope_max_dev, intercept_max_dev, r_max_dev, _, _ = linregress(distances, max_devs)
    pct_not_visable, intercept_visable_x, r_visable, _, _ = linregress(distances, visable)
    
    print(f"Std in X vs Distance:        std_x = {slope_std_x:.6f} * x + {intercept_std_x:.6f} (R² = {r_std_x**2:.4f})")
    print(f"Std in Y vs Distance:        std_y = {slope_std_y:.6f} * x + {intercept_std_y:.6f} (R² = {r_std_y**2:.4f})")
    print(f"Max Deviation vs Distance: max_dev = {slope_max_dev:.6f} * x + {intercept_max_dev:.6f} (R² = {r_max_dev**2:.4f})")
    print()
    if ANALYSIS_MODE == 'unobstructed':
        print(f"Note: Statistics calculated using only non-obstructed AND visible points")
    else:
        print(f"Pct not visable vs Distance: pct = {pct_not_visable:.6f} * x + {intercept_visable_x:.6f} (R² = {r_visable**2:.4f})")
        print(f"Note: Statistics calculated using only obstructed BUT visible points")
    print(f"      Max deviation measures the furthest point from mean position")
    print(f"      Std measures RMS deviation from mean position")
