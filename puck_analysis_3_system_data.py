import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def filter_and_analyze_puck_data(df, window_size=7, r_squared_threshold=0.98, max_residual=0.01, vel_threshold=0.3):
    """
    Filter air hockey puck data to remove collision points and calculate velocities.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns ['x', 'y', 'dt']
    window_size : int
        Number of points to use for local line fitting (should be odd)
    r_squared_threshold : float
        Minimum R² value for spatial (x,y) fit to be considered on a straight line
    max_residual : float
        Maximum acceptable perpendicular distance (in meters) from the spatial line
    
    Returns:
    --------
    pandas.DataFrame with columns ['x', 'y', 't', 'vx', 'vy']
        Filtered data with cumulative time and velocity components
    """
    # Calculate cumulative time
    t = np.cumsum(df['dt'].values)
    t = np.insert(t, 0, 0)[:-1]  # Shift so first point is at t=0
    
    # Create extended dataframe with time
    data = pd.DataFrame({
        'x': df['x'].values,
        'y': df['y'].values,
        't': t
    })
    
    n = len(data)
    half_window = window_size // 2
    
    # Arrays to store results
    keep_mask = np.zeros(n, dtype=bool)
    vx_array = np.zeros(n)
    vy_array = np.zeros(n)
    
    for i in range(n):
        # Define window around point i
        start_idx = max(0, i - half_window)
        end_idx = min(n, i + half_window + 1)
        
        # Get window data
        window_x = data['x'].iloc[start_idx:end_idx].values
        window_y = data['y'].iloc[start_idx:end_idx].values
        window_t = data['t'].iloc[start_idx:end_idx].values
        
        # Skip if window too small
        if len(window_x) < 3:
            continue
        
        # ===== STEP 1: Screen points by fitting a line through (x,y) spatial coordinates =====
        # Fit line y = mx + b through the (x,y) points
        # Handle vertical lines by checking variance
        var_x = np.var(window_x)
        var_y = np.var(window_y)
        
        if var_x > var_y:
            # Fit y = mx + b (more horizontal line)
            slope, intercept, r, _, _ = stats.linregress(window_x, window_y)
            # Calculate perpendicular distance from each point to the line
            # Line: mx - y + b = 0, distance = |mx - y + b| / sqrt(m^2 + 1)
            residuals = np.abs(slope * window_x - window_y + intercept) / np.sqrt(slope**2 + 1)
            r_squared = r ** 2
        else:
            # Fit x = my + b (more vertical line)
            slope, intercept, r, _, _ = stats.linregress(window_y, window_x)
            # Line: my - x + b = 0, distance = |my - x + b| / sqrt(m^2 + 1)
            residuals = np.abs(slope * window_y - window_x + intercept) / np.sqrt(slope**2 + 1)
            r_squared = r ** 2
        
        # Check if spatial line fits well with no outliers
        good_fit = r_squared >= r_squared_threshold
        no_outliers = np.max(residuals) <= max_residual
        
        if not (good_fit and no_outliers):
            continue  # Point is filtered out
        
        # ===== STEP 2: Calculate velocities from temporal fits =====
        # Now fit x vs t and y vs t to get velocity components
        slope_x, intercept_x, _, _, _ = stats.linregress(window_t, window_x)
        slope_y, intercept_y, _, _, _ = stats.linregress(window_t, window_y)
        
        # Store results
        if slope_x**2 + slope_y**2 > vel_threshold**2:
            keep_mask[i] = True
            vx_array[i] = slope_x
            vy_array[i] = slope_y
    
    # Create filtered dataframe
    filtered_df = pd.DataFrame({
        'x': data['x'][keep_mask].values,
        'y': data['y'][keep_mask].values,
        't': data['t'][keep_mask].values,
        'vx': vx_array[keep_mask],
        'vy': vy_array[keep_mask]
    })
    
    return filtered_df


def plot_data_segment(df, start_idx=None, end_idx=None, title="Air Hockey Puck Trajectory"):
    """
    Plot a segment of the trajectory data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns ['x', 'y']
    start_idx : int, optional
        Starting index for the segment (default: beginning)
    end_idx : int, optional
        Ending index for the segment (default: end)
    title : str
        Plot title
    """
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(df)
    
    # Extract segment
    segment = df.iloc[start_idx:end_idx]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot trajectory
    plt.plot(segment['x'], segment['y'], 'b-', linewidth=1, alpha=0.6, label='Trajectory')
    plt.plot(segment['x'], segment['y'], 'ro', markersize=3, alpha=0.7, label='Points')
    
    # Mark start and end
    plt.plot(segment['x'].iloc[0], segment['y'].iloc[0], 'go', 
             markersize=10, label='Start', zorder=5)
    plt.plot(segment['x'].iloc[-1], segment['y'].iloc[-1], 'rs', 
             markersize=10, label='End', zorder=5)
    
    # Draw table boundaries
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=1.9885, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=0.9905, color='k', linestyle='--', alpha=0.3)
    
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_filtered_comparison(original_df, filtered_df):
    """
    Plot original vs filtered data for comparison.
    
    Parameters:
    -----------
    original_df : pandas.DataFrame
        Original DataFrame with columns ['x', 'y']
    filtered_df : pandas.DataFrame
        Filtered DataFrame with columns ['x', 'y']
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Original data
    ax1.plot(original_df['x'], original_df['y'], 'b-', linewidth=1, alpha=0.4)
    ax1.plot(original_df['x'], original_df['y'], 'ro', markersize=2, alpha=0.5)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=1.9885, color='k', linestyle='--', alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axhline(y=0.9905, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title(f'Original Data ({len(original_df)} points)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Filtered data
    ax2.plot(filtered_df['x'], filtered_df['y'], 'g-', linewidth=1, alpha=0.6)
    ax2.plot(filtered_df['x'], filtered_df['y'], 'bo', markersize=3, alpha=0.7)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=1.9885, color='k', linestyle='--', alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axhline(y=0.9905, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('X Position (m)', fontsize=12)
    ax2.set_ylabel('Y Position (m)', fontsize=12)
    ax2.set_title(f'Filtered Data ({len(filtered_df)} points)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Points removed: {len(original_df) - len(filtered_df)} ({100*(1-len(filtered_df)/len(original_df)):.1f}%)")


def find_collision_points(filtered_df, wall_threshold=0.05, mallet_data=None, 
                          mallet_radius=0.1003/2, mallet_nearby_threshold=0.30,
                          mallet_time_window=0.1, poly_degree=3):
    """
    Find collision points by detecting velocity direction changes near walls or mallet.
    
    Parameters:
    -----------
    filtered_df : pandas.DataFrame
        Filtered DataFrame with columns ['x', 'y', 't', 'vx', 'vy']
    wall_threshold : float
        Maximum distance (in meters) from wall for collision to be valid
    mallet_data : pandas.DataFrame, optional
        DataFrame with columns ['Mx', 'My', 'Mvx', 'Mvy', 't'] for mallet tracking
        If None, only wall collisions are detected
    mallet_radius : float
        Radius of the mallet in meters
    mallet_nearby_threshold : float
        Distance threshold for mallet to be considered nearby
    mallet_time_window : float
        Time window (seconds) around collision to fit mallet path
    poly_degree : int
        Degree of polynomial for mallet path fitting
    
    Returns:
    --------
    list of dict
        Each dict contains collision information:
        - 'point_a_idx': index of point before collision
        - 'point_b_idx': index of point after collision
        - 'collision_x': x coordinate of collision point
        - 'collision_y': y coordinate of collision point
        - 'collision_time': time of collision
        - 'type': 'wall' or 'mallet'
        - 'wall': which wall ('left', 'right', 'top', 'bottom') if type='wall'
        - 'mallet_x', 'mallet_y': mallet position if type='mallet'
        - 'mallet_vx', 'mallet_vy': mallet velocity if type='mallet'
        - 'vx_pre': x velocity before collision
        - 'vy_pre': y velocity before collision
        - 'vx_post': x velocity after collision
        - 'vy_post': y velocity after collision
        - 'e_normal': normal restitution coefficient
        - 'e_tangential': tangential restitution coefficient
    """
    # Table boundaries
    x_min, x_max = 0, 1.9885
    y_min, y_max = 0, 0.9905
    puck_radius = 0.0309  # 6.18 cm diameter / 2
    
    collisions = []
    
    for i in range(len(filtered_df) - 1):
        # Get adjacent points
        row_a = filtered_df.iloc[i]
        row_b = filtered_df.iloc[i + 1]
        
        # Extract velocities
        v_a = np.array([row_a['vx'], row_a['vy']])
        v_b = np.array([row_b['vx'], row_b['vy']])
        
        # Check if velocities are in different directions (dot product < 0)
        dot_product = np.dot(v_a, v_b)
        if dot_product >= 0:
            continue  # Velocities not in opposite directions
        
        # Get positions
        pos_a = np.array([row_a['x'], row_a['y']])
        pos_b = np.array([row_b['x'], row_b['y']])
        
        # Find intersection of two lines defined by position + velocity direction
        # Line A: P_a + t * v_a
        # Line B: P_b + s * v_b
        # Solve: P_a + t * v_a = P_b + s * v_b
        # Rearranged: [v_a, -v_b] * [t, s]^T = P_b - P_a
        
        A_matrix = np.column_stack([v_a, -v_b])
        b_vector = pos_b - pos_a
        
        # Check if lines are parallel (determinant near zero)
        det = np.linalg.det(A_matrix)
        if abs(det) < 1e-10:
            continue  # Lines are parallel, no intersection
        
        # Solve for parameters
        try:
            params = np.linalg.solve(A_matrix, b_vector)
            t = params[0]
            s = params[1]
        except np.linalg.LinAlgError:
            continue  # Singular matrix
        
        # Calculate intersection point
        intersection = pos_a + t * v_a
        x_int, y_int = intersection
        
        # Estimate collision time (average of the two adjacent points)
        collision_time = (row_a['t'] + row_b['t']) / 2
        
        # Determine which wall (if any) the collision is near
        wall = None
        wall_normal = None
        
        dist_to_left = abs(x_int - x_min)
        dist_to_right = abs(x_int - x_max)
        dist_to_bottom = abs(y_int - y_min)
        dist_to_top = abs(y_int - y_max)
        
        min_dist = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
        
        if min_dist > wall_threshold or min_dist < 0.01:
            # Not near any wall - check for mallet collision
            if mallet_data is None:
                print("No mallet data provided")
                continue  # No mallet data provided
            
            # Find mallet data points within time window around collision
            time_mask = np.abs(mallet_data['t'] - collision_time) <= mallet_time_window
            
            if np.sum(time_mask) < poly_degree + 2:
                print("Not enough points to fit polynomial")
                continue  # Not enough points to fit polynomial
            
            mallet_t = mallet_data['t'][time_mask].values
            mallet_x = mallet_data['Mx'][time_mask].values
            mallet_y = mallet_data['My'][time_mask].values
            
            # Quick check: is mallet ever nearby?
            distances = np.sqrt((mallet_x - x_int)**2 + (mallet_y - y_int)**2)
            if np.min(distances) > mallet_nearby_threshold:
                print("Mallet never gets close enough")
                continue  # Mallet never gets close enough
            
            # Fit polynomial to mallet path
            try:
                poly_x = np.polyfit(mallet_t, mallet_x, poly_degree)
                poly_y = np.polyfit(mallet_t, mallet_y, poly_degree)
            except (np.linalg.LinAlgError, np.RankWarning):
                print("fit error")
                continue
            
            # Find point on mallet path that minimizes |distance_to_collision - (mallet_radius + puck_radius)|
            # Sample the polynomial at fine time resolution
            t_fine = np.linspace(mallet_t.min(), mallet_t.max(), 500)
            x_fine = np.polyval(poly_x, t_fine)
            y_fine = np.polyval(poly_y, t_fine)
            
            # Distance from mallet center to collision point
            dist_to_collision = np.sqrt((x_fine - x_int)**2 + (y_fine - y_int)**2)
            
            # Error: |distance - (mallet_radius + puck_radius)|
            contact_distance = mallet_radius + puck_radius
            error = np.abs(dist_to_collision - contact_distance)
            
            # Find points that minimize error
            min_error = np.min(error)
            if min_error > 0.05:  # If error too large, not a valid collision
                print("err too large")
                continue
            
            candidate_indices = np.where(error < 0.04)[0]
            
            if len(candidate_indices) == 0:
                print("no canidates")
                continue
            
            # Among candidates, choose the one most aligned with incoming puck direction
            best_idx = None
            best_alignment = -np.inf
            
            for idx in candidate_indices:
                # Direction from collision point to mallet center
                collision_to_mallet = np.array([x_fine[idx] - x_int, y_fine[idx] - y_int])
                collision_to_mallet_norm = collision_to_mallet / (np.linalg.norm(collision_to_mallet) + 1e-10)
                
                # Change in puck velocity
                delta_v = v_b - v_a
                delta_v_norm = delta_v / (np.linalg.norm(delta_v) + 1e-10)
                
                # Check if collision_to_mallet is opposite to delta_v (dot product should be negative)
                alignment = np.dot(collision_to_mallet_norm, delta_v_norm)
                
                if alignment < 0:  # Opposite directions (as expected for collision)
                    # Among valid candidates, choose one with least distance error
                    if best_idx is None or error[idx] < error[best_idx]:
                        best_idx = idx
            
            if best_idx is None:
                print("no best idx")
                continue
            
            # Get mallet state at contact
            t_contact = t_fine[best_idx]
            mallet_x_contact = x_fine[best_idx]
            mallet_y_contact = y_fine[best_idx]
            
            # Calculate mallet velocity at contact (derivative of polynomial)
            poly_x_deriv = np.polyder(poly_x)
            poly_y_deriv = np.polyder(poly_y)
            
            mallet_vx_contact = np.polyval(poly_x_deriv, t_contact)
            mallet_vy_contact = np.polyval(poly_y_deriv, t_contact)
            
            # Normal direction: from mallet center to collision point
            normal_vec = np.array([x_int - mallet_x_contact, y_int - mallet_y_contact])
            normal_vec = normal_vec / np.linalg.norm(normal_vec)
            
            # Tangent direction: perpendicular to normal
            tangent_vec = np.array([-normal_vec[1], normal_vec[0]])
            
            # Get puck velocities in mallet frame (puck velocity - mallet velocity)
            v_rel_before = v_a - np.array([mallet_vx_contact, mallet_vy_contact])
            v_rel_after = v_b - np.array([mallet_vx_contact, mallet_vy_contact])
            
            # Decompose relative velocities into normal and tangential components
            v_normal_pre = np.dot(v_rel_before, normal_vec)
            v_tangent_pre = np.dot(v_rel_before, tangent_vec)
            
            v_normal_post = np.dot(v_rel_after, normal_vec)
            v_tangent_post = np.dot(v_rel_after, tangent_vec)
            
            # Calculate restitution coefficients
            if abs(v_normal_pre) < 1e-6:
                print("v normal pre is 0")
                continue  # Avoid division by zero
            
            e_normal = -v_normal_post / v_normal_pre
            
            if abs(v_tangent_pre) < 1e-6:
                print("v tangent pre is 0")
                e_tangential = 1.0  # No tangential component, assume elastic
            else:
                e_tangential = v_tangent_post / v_tangent_pre
            
            if True: #e_tangential < 1.3 and e_normal < 1.3:
                # Record mallet collision
                collision_data = {
                    'point_a_idx': i,
                    'point_b_idx': i + 1,
                    'collision_x': x_int,
                    'collision_y': y_int,
                    'collision_time': collision_time,
                    'type': 'mallet',
                    'mallet_x': mallet_x_contact,
                    'mallet_y': mallet_y_contact,
                    'mallet_vx': mallet_vx_contact,
                    'mallet_vy': mallet_vy_contact,
                    'vx_pre': row_a['vx'],
                    'vy_pre': row_a['vy'],
                    'vx_post': row_b['vx'],
                    'vy_post': row_b['vy'],
                    'e_normal': e_normal,
                    'e_tangential': e_tangential,
                    'v_normal_pre': v_normal_pre,
                    'v_tangent_pre': v_tangent_pre,
                    'v_normal_post': v_normal_post,
                    'v_tangent_post': v_tangent_post
                }
                collisions.append(collision_data)
            else:
                print("restititution out of bounds")
            
            
    
    return collisions


def plot_collisions(filtered_df, collisions, mallet_data=None, puck_radius=0.0309, mallet_radius=0.0475):
    """
    Visualize detected collisions on the trajectory.
    
    Parameters:
    -----------
    filtered_df : pandas.DataFrame
        Filtered DataFrame with columns ['x', 'y']
    collisions : list of dict
        List of collision data from find_collision_points
    mallet_data : pandas.DataFrame, optional
        DataFrame with mallet positions for visualization
    puck_radius : float
        Radius of puck
    mallet_radius : float
        Radius of mallet
    """
    plt.figure(figsize=(14, 7))
    
    # Plot trajectory
    plt.plot(filtered_df['x'], filtered_df['y'], 'b-', linewidth=1, alpha=0.4, label='Puck Trajectory')
    plt.plot(filtered_df['x'], filtered_df['y'], 'bo', markersize=2, alpha=0.3)
    
    # Plot mallet trajectory if available
    if mallet_data is not None:
        plt.plot(mallet_data['Mx'], mallet_data['My'], 'g-', linewidth=1, alpha=0.3, label='Mallet Trajectory')
    
    # Plot collision points
    wall_collisions = [c for c in collisions if c['type'] == 'wall']
    mallet_collisions = [c for c in collisions if c['type'] == 'mallet']
    
    for collision in wall_collisions:
        x_int = collision['collision_x']
        y_int = collision['collision_y']
        plt.plot(x_int, y_int, 'r*', markersize=15, markeredgecolor='black', 
                markeredgewidth=0.5, label='Wall Collision' if collision == wall_collisions[0] else '')
        
        # Annotate with wall name
        plt.annotate(collision['wall'], (x_int, y_int), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    for collision in mallet_collisions:
        x_int = collision['collision_x']
        y_int = collision['collision_y']
        mx = collision['mallet_x']
        my = collision['mallet_y']
        
        plt.plot(x_int, y_int, 'm*', markersize=15, markeredgecolor='black',
                markeredgewidth=0.5, label='Mallet Collision' if collision == mallet_collisions[0] else '')
        
        # Draw puck at collision
        circle_puck = plt.Circle((x_int, y_int), puck_radius, color='blue', 
                                 fill=False, linewidth=2, alpha=0.6)
        plt.gca().add_patch(circle_puck)
        
        # Draw mallet at collision
        circle_mallet = plt.Circle((mx, my), mallet_radius, color='green',
                                   fill=False, linewidth=2, alpha=0.6)
        plt.gca().add_patch(circle_mallet)
        
        # Draw line between centers
        plt.plot([x_int, mx], [y_int, my], 'k--', linewidth=1, alpha=0.3)
    
    # Draw table boundaries
    plt.axvline(x=0, color='k', linestyle='--', linewidth=2, alpha=0.5)
    plt.axvline(x=1.9885, color='k', linestyle='--', linewidth=2, alpha=0.5)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.5)
    plt.axhline(y=0.9905, color='k', linestyle='--', linewidth=2, alpha=0.5)
    
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.title(f'Detected Collisions ({len(wall_collisions)} wall, {len(mallet_collisions)} mallet)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    # Print collision summary
    print(f"\nFound {len(wall_collisions)} wall collisions and {len(mallet_collisions)} mallet collisions:")
    
    if wall_collisions:
        print("\n--- Wall Collisions ---")
        for i, collision in enumerate(wall_collisions):
            print(f"\nWall Collision {i+1}:")
            print(f"  Wall: {collision['wall']}")
            print(f"  Location: ({collision['collision_x']:.4f}, {collision['collision_y']:.4f})")
            print(f"  Pre-collision velocity: ({collision['vx_pre']:.3f}, {collision['vy_pre']:.3f}) m/s")
            print(f"  Post-collision velocity: ({collision['vx_post']:.3f}, {collision['vy_post']:.3f}) m/s")
            print(f"  Normal restitution (e_n): {collision['e_normal']:.3f}")
            print(f"  Tangential restitution (e_t): {collision['e_tangential']:.3f}")
    
    if mallet_collisions:
        print("\n--- Mallet Collisions ---")
        for i, collision in enumerate(mallet_collisions):
            print(f"\nMallet Collision {i+1}:")
            print(f"  Time: {collision['collision_time']:.3f} s")
            print(f"  Location: ({collision['collision_x']:.4f}, {collision['collision_y']:.4f})")
            print(f"  Puck velocity (pre): ({collision['vx_pre']:.3f}, {collision['vy_pre']:.3f}) m/s")
            print(f"  Puck velocity (post): ({collision['vx_post']:.3f}, {collision['vy_post']:.3f}) m/s")
            print(f"  Mallet velocity: ({collision['mallet_vx']:.3f}, {collision['mallet_vy']:.3f}) m/s")
            print(f"  Relative normal velocity: {collision['v_normal_pre']:.3f} m/s")
            print(f"  Relative tangent velocity: {collision['v_tangent_pre']:.3f} m/s")
            print(f"  Normal restitution (e_n): {collision['e_normal']:.3f}")
            print(f"  Tangential restitution (e_t): {collision['e_tangential']:.3f}")


def restitution_model(v, n_f, n_0, n_r):
    """
    Restitution coefficient model as a function of velocity.
    
    e = n_f + (1 - n_f/n_0) * 2/(1 + exp(n_r * v^2))
    
    Parameters:
    -----------
    v : float or array
        Velocity magnitude
    n_f : float
        Final restitution at high velocity
    n_0 : float
        Initial restitution parameter
    n_r : float
        Rate parameter
    
    Returns:
    --------
    float or array
        Restitution coefficient
    """
    return n_f + (1 - n_f/n_0) * 2 / (1 + np.exp(n_r * v**2))


def fit_restitution_model(collisions):
    """
    Fit restitution models for normal and tangential coefficients.
    
    Parameters:
    -----------
    collisions : list of dict
        List of collision data from find_collision_points
    
    Returns:
    --------
    dict with keys:
        'normal_params': (n_f, n_0, n_r) for normal restitution
        'tangential_params': (n_f, n_0, n_r) for tangential restitution
        'sigma': standard deviation of Gaussian noise
        'normal_velocities': array of normal velocities
        'tangential_velocities': array of tangential velocities
        'normal_restitutions': array of normal restitution coefficients
        'tangential_restitutions': array of tangential restitution coefficients
    """
    from scipy.optimize import curve_fit, minimize
    
    # Extract data
    normal_vels = np.array([abs(c['v_normal_pre']) for c in collisions])
    tangent_vels = np.array([abs(c['v_tangent_pre']) for c in collisions])
    normal_rest = np.array([c['e_normal'] for c in collisions])
    tangent_rest = np.array([c['e_tangential'] for c in collisions])
    
    # Fit normal restitution model
    try:
        # Initial guess: n_f=0.5, n_0=1.0, n_r=0.1
        normal_params, _ = curve_fit(
            restitution_model, 
            normal_vels, 
            normal_rest,
            p0=[0.5, 1.0, 0.1],
            bounds=([0, 0.1, 0], [1, 2, 10]),
            maxfev=10000
        )
    except Exception as e:
        print(f"Warning: Normal fit failed, using default parameters. Error: {e}")
        normal_params = [0.5, 1.0, 0.1]
    
    # Fit tangential restitution model
    try:
        tangent_params, _ = curve_fit(
            restitution_model,
            tangent_vels,
            tangent_rest,
            p0=[0.5, 1.0, 0.1],
            bounds=([0, 0.1, 0], [1, 2, 10]),
            maxfev=10000
        )
    except Exception as e:
        print(f"Warning: Tangential fit failed, using default parameters. Error: {e}")
        tangent_params = [0.5, 1.0, 0.1]
    
    # Estimate sigma separately for normal and tangential
    normal_predicted = restitution_model(normal_vels, *normal_params)
    tangent_predicted = restitution_model(tangent_vels, *tangent_params)
    
    # Compute residuals separately
    normal_residuals = normal_rest - normal_predicted
    tangent_residuals = tangent_rest - tangent_predicted
    
    # Simple MLE estimates (sample standard deviation)
    sigma_normal_simple = np.std(normal_residuals, ddof=1)
    sigma_tangent_simple = np.std(tangent_residuals, ddof=1)
    
    # Optimize for sigma using maximum likelihood
    def negative_log_likelihood(sigmas):
        sigma_n, sigma_t = sigmas
        if sigma_n <= 0 or sigma_t <= 0:
            return 1e10
        # Log likelihood for Gaussian (separate sigmas)
        normal_ll = -0.5 * np.sum((normal_residuals / sigma_n)**2) - len(normal_rest) * np.log(sigma_n * np.sqrt(2 * np.pi))
        tangent_ll = -0.5 * np.sum((tangent_residuals / sigma_t)**2) - len(tangent_rest) * np.log(sigma_t * np.sqrt(2 * np.pi))
        return -(normal_ll + tangent_ll)
    
    # Optimize for both sigmas
    result = minimize(
        negative_log_likelihood, 
        x0=[sigma_normal_simple, sigma_tangent_simple], 
        bounds=[(1e-6, 1), (1e-6, 1)], 
        method='L-BFGS-B'
    )
    sigma_normal_mle, sigma_tangent_mle = result.x
    
    return {
        'normal_params': normal_params,
        'tangential_params': tangent_params,
        'sigma_normal': sigma_normal_mle,
        'sigma_tangent': sigma_tangent_mle,
        'sigma_normal_simple': sigma_normal_simple,
        'sigma_tangent_simple': sigma_tangent_simple,
        'normal_velocities': normal_vels,
        'tangential_velocities': tangent_vels,
        'normal_restitutions': normal_rest,
        'tangential_restitutions': tangent_rest
    }


def plot_restitution_analysis(collisions, fit_results):
    """
    Plot restitution coefficients vs velocities with fitted models.
    
    Parameters:
    -----------
    collisions : list of dict
        List of collision data
    fit_results : dict
        Results from fit_restitution_model
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Extract data
    normal_vels = fit_results['normal_velocities']
    tangent_vels = fit_results['tangential_velocities']
    normal_rest = fit_results['normal_restitutions']
    tangent_rest = fit_results['tangential_restitutions']
    
    normal_params = fit_results['normal_params']
    tangent_params = fit_results['tangential_params']
    sigma_normal = fit_results['sigma_normal']
    sigma_tangent = fit_results['sigma_tangent']
    
    # Generate smooth curves for plotting
    v_normal_range = np.linspace(0, max(normal_vels) * 1.1, 200)
    v_tangent_range = np.linspace(0, max(tangent_vels) * 1.1, 200)
    
    normal_fitted = restitution_model(v_normal_range, *normal_params)
    tangent_fitted = restitution_model(v_tangent_range, *tangent_params)
    
    # Plot normal restitution
    ax1.scatter(normal_vels, normal_rest, alpha=0.6, s=50, color='blue', label='Data')
    ax1.plot(v_normal_range, normal_fitted, 'r-', linewidth=2, label='Fitted Model')
    
    # Add uncertainty band (±sigma_normal)
    normal_fitted_points = restitution_model(normal_vels, *normal_params)
    ax1.fill_between(v_normal_range, 
                     restitution_model(v_normal_range, *normal_params) - sigma_normal,
                     restitution_model(v_normal_range, *normal_params) + sigma_normal,
                     alpha=0.2, color='red', label=f'±σ_n = ±{sigma_normal:.3f}')
    
    ax1.set_xlabel('Normal Velocity |v_n| (m/s)', fontsize=12)
    ax1.set_ylabel('Normal Restitution e_n', fontsize=12)
    ax1.set_title(f'Normal Restitution vs Velocity\nn_f={normal_params[0]:.3f}, n_0={normal_params[1]:.3f}, n_r={normal_params[2]:.3f}', 
                  fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, max(1.2, max(normal_rest) * 1.1)])
    
    # Plot tangential restitution
    ax2.scatter(tangent_vels, tangent_rest, alpha=0.6, s=50, color='green', label='Data')
    ax2.plot(v_tangent_range, tangent_fitted, 'r-', linewidth=2, label='Fitted Model')
    
    # Add uncertainty band (±sigma_tangent)
    ax2.fill_between(v_tangent_range,
                     restitution_model(v_tangent_range, *tangent_params) - sigma_tangent,
                     restitution_model(v_tangent_range, *tangent_params) + sigma_tangent,
                     alpha=0.2, color='red', label=f'±σ_t = ±{sigma_tangent:.3f}')
    
    ax2.set_xlabel('Tangential Velocity |v_t| (m/s)', fontsize=12)
    ax2.set_ylabel('Tangential Restitution e_t', fontsize=12)
    ax2.set_title(f'Tangential Restitution vs Velocity\nn_f={tangent_params[0]:.3f}, n_0={tangent_params[1]:.3f}, n_r={tangent_params[2]:.3f}',
                  fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, max(1.2, max(tangent_rest) * 1.1)])
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("\n" + "="*60)
    print("RESTITUTION MODEL FITTING RESULTS")
    print("="*60)
    print(f"\nNormal Restitution Parameters:")
    print(f"  n_f = {normal_params[0]:.4f}")
    print(f"  n_0 = {normal_params[1]:.4f}")
    print(f"  n_r = {normal_params[2]:.4f}")
    print(f"\nTangential Restitution Parameters:")
    print(f"  n_f = {tangent_params[0]:.4f}")
    print(f"  n_0 = {tangent_params[1]:.4f}")
    print(f"  n_r = {tangent_params[2]:.4f}")
    print(f"\nGaussian Noise Standard Deviations:")
    print(f"  σ_normal (MLE) = {sigma_normal:.4f}")
    print(f"  σ_normal (simple) = {fit_results['sigma_normal_simple']:.4f}")
    print(f"  σ_tangent (MLE) = {sigma_tangent:.4f}")
    print(f"  σ_tangent (simple) = {fit_results['sigma_tangent_simple']:.4f}")
    print(f"\nNumber of collisions analyzed: {len(collisions)}")
    print("="*60)


# --- For puck-mallet collision analysis ---
# Load system loop data with mallet tracking
df_system = pd.read_csv('system_loop_data.csv')

# Calculate cumulative time for mallet data
mallet_t = np.cumsum(df_system['dt'].values)
mallet_t = np.insert(mallet_t, 0, 0)[:-1]

mallet_data = pd.DataFrame({
    'Mx': df_system['Mx'].values,
    'My': df_system['My'].values,
    'Mvx': df_system['Mxv'].values,
    'Mvy': df_system['Myv'].values,
    't': mallet_t
})

# Extract puck data
puck_df = pd.DataFrame({
    'x': df_system['Px'].values,
    'y': df_system['Py'].values,
    'dt': df_system['dt'].values
})

# Filter puck data
filtered_puck_df = filter_and_analyze_puck_data(
    puck_df,
    window_size=7,
    r_squared_threshold=0.98,
    max_residual=0.01
)

# Find collisions (both wall and mallet)
all_collisions = find_collision_points(
    filtered_puck_df, 
    wall_threshold=0.05,
    mallet_data=mallet_data,
    mallet_radius=0.1003/2,
    mallet_nearby_threshold=0.40,
    mallet_time_window=0.05,
    poly_degree=3
)

# Visualize
plot_collisions(filtered_puck_df, all_collisions, mallet_data=mallet_data)

# Fit restitution models using both wall and mallet collisions
fit_results = fit_restitution_model(all_collisions)

# Plot restitution analysis
plot_restitution_analysis(all_collisions, fit_results)

# Access fitted parameters
print(f"Normal params: {fit_results['normal_params']}")
print(f"Tangential params: {fit_results['tangential_params']}")
print(f"Sigma normal: {fit_results['sigma_normal']}")
print(f"Sigma tangent: {fit_results['sigma_tangent']}")
