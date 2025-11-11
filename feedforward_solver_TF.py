import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize, differential_evolution
from scipy.signal import savgol_filter

# Read the CSV file
df = pd.read_csv('feedforward_data_12V_Supercap.csv')

# Remove the second row (index 1) which contains zeros
df = df.drop(index=1).reset_index(drop=True)

# Extract data
x_measured = df['x'].values
y_measured = df['y'].values
left_pwm = df['Left_PWM'].values
right_pwm = df['Right_PWM'].values
dt_data = df['dt'].values

# Calculate Vx
Vx = (left_pwm - right_pwm) * 24

# Calculate time array (cumulative sum of dt)
t = np.concatenate([[0], np.cumsum(dt_data[:-1])])

# Apply smoothing to reduce noise
window_size = 61
poly_order = 3
x_smooth = x_measured #savgol_filter(x_measured, window_size, poly_order)
Vx_smooth = Vx #savgol_filter(Vx, window_size, poly_order)

def get_initial_conditions(x_data, t_data, window=21, poly_order=5):
    """
    Get initial conditions x(0), x'(0), x''(0) using polynomial fit
    Takes the center point of the window as the IC
    """
    # Use first 'window' points
    t_window = t_data[:window]
    x_window = x_data[:window]
    
    # Fit polynomial
    coeffs = np.polyfit(t_window, x_window, poly_order)
    
    # Get derivatives at center of window
    center_idx = window // 2
    t_center = t_data[center_idx]
    
    deriv1_coeffs = np.polyder(coeffs, 1)
    deriv2_coeffs = np.polyder(coeffs, 2)
    
    x0 = np.polyval(coeffs, t_center)
    x_dot0 = np.polyval(deriv1_coeffs, t_center)
    x_ddot0 = np.polyval(deriv2_coeffs, t_center)
    
    return x0, x_dot0, x_ddot0, center_idx

def simulate_system(coeffs, t_sim, Vx_func, initial_conditions):
    """
    Simulate the system: Vx = a*x''' + b*x'' + c*x'
    Rewrite as state space: [x, x', x''] with x''' = (Vx - b*x'' - c*x')/a
    
    Args:
        coeffs: [a, b, c]
        t_sim: time array
        Vx_func: function that returns Vx at time t
        initial_conditions: [x0, x_dot0, x_ddot0]
    """
    a, b, c = coeffs
    
    # Avoid division by very small a
    if abs(a) < 1e-12:
        a = 1e-12 * np.sign(a) if a != 0 else 1e-12
    
    def state_derivative(state, t):
        """
        state = [x, x', x'']
        derivatives = [x', x'', x''']
        where x''' = (Vx - b*x'' - c*x')/a
        """
        x, x_dot, x_ddot = state
        
        # Get Vx at current time
        Vx_t = Vx_func(t)
        
        # Calculate x'''
        x_dddot = (Vx_t - b * x_ddot - c * x_dot) / a
        
        return [x_dot, x_ddot, x_dddot]
    
    # Integrate
    solution = odeint(state_derivative, initial_conditions, t_sim)
    
    return solution[:, 0]  # Return x(t)

def split_data_into_segments(t, x, Vx, segment_length):
    """
    Split data into overlapping segments
    """
    segments = []
    n = len(t)
    
    # Create segments with 50% overlap
    step = segment_length // 2
    
    for i in range(0, n - segment_length + 1, step):
        end_idx = i + segment_length
        
        # Get segment
        t_seg = t[i:end_idx] - t[i]  # Normalize time to start at 0
        x_seg = x[i:end_idx]
        Vx_seg = Vx[i:end_idx]
        
        segments.append({
            't': t_seg,
            'x': x_seg,
            'Vx': Vx_seg,
            'start_idx': i
        })
    
    return segments

def objective_function(coeffs, segment):
    """
    Objective: minimize error between simulated and measured x
    """
    t_seg = segment['t']
    x_measured_seg = segment['x']
    Vx_seg = segment['Vx']
    
    # Create interpolation function for Vx
    Vx_func = lambda t: np.interp(t, t_seg, Vx_seg)
    
    # Get initial conditions at center of window
    x0, x_dot0, x_ddot0, center_idx = get_initial_conditions(x_measured_seg, t_seg)
    initial_conditions = [x0, x_dot0, x_ddot0]
    
    # Simulate from center point forward
    t_sim = t_seg[center_idx:]
    
    try:
        # Simulate
        x_simulated = simulate_system(coeffs, t_sim, Vx_func, initial_conditions)
        
        # Calculate error (mean squared error) only for simulated portion
        x_measured_portion = x_measured_seg[center_idx:]
        error = np.mean((x_measured_portion - x_simulated)**2)
        
        return error
    except:
        # Return large error if simulation fails
        return 1e10

def optimize_coefficients(segments, method='global'):
    """
    Optimize coefficients a, b, c across all segments
    """
    def total_objective(coeffs):
        """Sum of errors across all segments"""
        total_error = 0
        for segment in segments:
            error = objective_function(coeffs, segment)
            total_error += error
        return total_error
    
    # Bounds for coefficients (adjust as needed)
    bounds = [(-0.001, 0.001), (-0.1, 0.5), (0.1, 5)]
    
    print(f"Optimizing over {len(segments)} segments...")
    
    if method == 'global':
        # Use differential evolution for global optimization
        result = differential_evolution(
            total_objective,
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            workers=1,
            disp=True,
            seed=42
        )
    else:
        # Use local optimization with multiple starting points
        x0_list = [
            [1.0, 1.0, 1.0],
            [0.1, 0.1, 0.1],
            [10.0, 10.0, 10.0],
            [-1.0, -1.0, -1.0]
        ]
        
        best_result = None
        best_error = float('inf')
        
        for x0 in x0_list:
            result = minimize(
                total_objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            if result.fun < best_error:
                best_error = result.fun
                best_result = result
        
        result = best_result
    
    return result

# Choose segment length (adjust based on your data characteristics)
segment_length = min(500, len(t) // 4)  # Use segments of ~200 points or 1/4 of data
print(f"Using segment length: {segment_length} points")

# Split data into segments
segments = split_data_into_segments(t, x_smooth, Vx_smooth, segment_length)
print(f"Created {len(segments)} segments")

# Optimize coefficients
result = optimize_coefficients(segments, method='global')

a_opt, b_opt, c_opt = result.x

print("\n" + "="*60)
print("OPTIMIZED COEFFICIENTS:")
print("="*60)
print(f"a = {a_opt:.6e}")
print(f"b = {b_opt:.6e}")
print(f"c = {c_opt:.6e}")
print(f"\nDifferential equation: Vx = {a_opt:.3e}·x''' + {b_opt:.3e}·x'' + {c_opt:.3e}·x'")
print(f"Transfer function: X(s)/Vx(s) = 1/({a_opt:.3e}·s³ + {b_opt:.3e}·s² + {c_opt:.3e}·s)")
print(f"\nTotal optimization error: {result.fun:.6e}")
print("="*60)

# Simulate full trajectory with optimized coefficients
Vx_func_full = lambda t_val: np.interp(t_val, t, Vx_smooth)
x0, x_dot0, x_ddot0, center_idx = get_initial_conditions(x_smooth, t)
initial_conditions = [x0, x_dot0, x_ddot0]

# Simulate from center point forward
t_sim_full = t[center_idx:]
x_simulated_partial = simulate_system([a_opt, b_opt, c_opt], t_sim_full, Vx_func_full, initial_conditions)

# Create full trajectory array (pad beginning with NaN)
x_simulated_full = np.full(len(t), np.nan)
x_simulated_full[center_idx:] = x_simulated_partial

# Calculate goodness of fit (only for simulated portion)
x_smooth_valid = x_smooth[center_idx:]
ss_res = np.sum((x_smooth_valid - x_simulated_partial)**2)
ss_tot = np.sum((x_smooth_valid - np.mean(x_smooth_valid))**2)
r_squared = 1 - (ss_res / ss_tot)

print(f"\nFull trajectory R²: {r_squared:.6f}")
print(f"RMSE: {np.sqrt(np.mean((x_smooth_valid - x_simulated_partial)**2)):.6f}")

# Plotting
fig, axes = plt.subplots(3, 2, figsize=(16, 12))

# Plot 1: Full trajectory comparison
axes[0, 0].plot(t, x_measured, 'b-', alpha=0.3, label='X measured (raw)', linewidth=1)
axes[0, 0].plot(t, x_smooth, 'b-', alpha=0.7, label='X measured (smoothed)', linewidth=2)
axes[0, 0].plot(t, x_simulated_full, 'r--', label='X simulated', linewidth=2)
axes[0, 0].axvline(x=t[center_idx], color='g', linestyle=':', linewidth=2, label='IC point')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('X position')
axes[0, 0].set_title(f'Full Trajectory Comparison (R² = {r_squared:.4f})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Residuals
residuals = np.full(len(t), np.nan)
residuals[center_idx:] = x_smooth[center_idx:] - x_simulated_partial
axes[0, 1].plot(t, residuals, 'g-', alpha=0.7)
axes[0, 1].axhline(y=0, color='k', linestyle='--', linewidth=1)
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Residual')
axes[0, 1].set_title('Residuals (Measured - Simulated)')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Vx input signal
axes[1, 0].plot(t, Vx, 'c-', alpha=0.3, label='Vx (raw)', linewidth=1)
axes[1, 0].plot(t, Vx_smooth, 'c-', alpha=0.7, label='Vx (smoothed)', linewidth=2)
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Vx (PWM difference)')
axes[1, 0].set_title('Input Signal Vx')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Segment-wise errors
segment_errors = [objective_function([a_opt, b_opt, c_opt], seg) for seg in segments]
segment_centers = [seg['start_idx'] + len(seg['t'])//2 for seg in segments]
segment_times = [t[idx] for idx in segment_centers]
axes[1, 1].plot(segment_times, segment_errors, 'mo-', linewidth=2, markersize=6)
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Segment MSE')
axes[1, 1].set_title('Error by Segment')
axes[1, 1].grid(True, alpha=0.3)

# Plot 5: Zoom into a segment
zoom_seg_idx = len(segments) // 2  # Middle segment
zoom_seg = segments[zoom_seg_idx]
Vx_func_zoom = lambda t_val: np.interp(t_val, zoom_seg['t'], zoom_seg['Vx'])
x0_zoom, x_dot0_zoom, x_ddot0_zoom, center_idx_zoom = get_initial_conditions(zoom_seg['x'], zoom_seg['t'])
t_zoom_sim = zoom_seg['t'][center_idx_zoom:]
x_sim_zoom = simulate_system([a_opt, b_opt, c_opt], t_zoom_sim, Vx_func_zoom, 
                              [x0_zoom, x_dot0_zoom, x_ddot0_zoom])
axes[2, 0].plot(zoom_seg['t'], zoom_seg['x'], 'b-', linewidth=2, label='Measured')
x_sim_full_zoom = np.full(len(zoom_seg['t']), np.nan)
x_sim_full_zoom[center_idx_zoom:] = x_sim_zoom
axes[2, 0].plot(zoom_seg['t'], x_sim_full_zoom, 'r--', linewidth=2, label='Simulated')
axes[2, 0].axvline(x=zoom_seg['t'][center_idx_zoom], color='g', linestyle=':', linewidth=2, label='IC point')
axes[2, 0].set_xlabel('Time (s)')
axes[2, 0].set_ylabel('X position')
axes[2, 0].set_title(f'Zoom: Segment {zoom_seg_idx+1} of {len(segments)}')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# Plot 6: Histogram of residuals
residuals_valid = residuals[~np.isnan(residuals)]
axes[2, 1].hist(residuals_valid, bins=50, alpha=0.7, color='purple', edgecolor='black')
axes[2, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[2, 1].set_xlabel('Residual value')
axes[2, 1].set_ylabel('Frequency')
axes[2, 1].set_title(f'Residual Distribution (μ={np.mean(residuals_valid):.4f}, σ={np.std(residuals_valid):.4f})')
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('time_domain_optimization_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nPlot saved as 'time_domain_optimization_results.png'")