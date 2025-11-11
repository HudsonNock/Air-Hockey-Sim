
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler

"""
Air hockey trajectory segmentation and wall-collision detection

This module provides functions to:
- load (x,y,dt) CSV data and compute timestamps and velocities
- segment the trajectory into linear, reasonably-fast motion intervals
  using a greedy window-growing algorithm with PCA-based linearity check
- for adjacent linear segments, compute the intersection point of their
  fitted lines and decide whether it corresponds to a wall collision
- compute normal and tangential restitution coefficients from pre/post
  collision velocities
- plotting helpers for visual verification

Usage example (from command line / notebook):

from air_hockey_collision_segmentation import (
    load_data, compute_timestamps_and_velocity, segment_linear_greedy,
    detect_collisions, plot_segments_and_collisions
)

df = load_data('puck_data.csv')
df = compute_timestamps_and_velocity(df)
segments = segment_linear_greedy(df)
collisions = detect_collisions(df, segments)
plot_segments_and_collisions(df, segments, collisions)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow.keras.backend as K

def filter_and_analyze_puck_data(df, window_size=7, r_squared_threshold=0.9, max_residual=0.01, vel_threshold=0.1):
    """
    Filter air hockey puck data to remove collision points and calculate velocities.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns ['x', 'y', 'dt']
    window_size : int
        Number of points to use for local line fitting (should be odd)
    r_squared_threshold : float
        Minimum R² value for a point to be considered on a straight line
    max_residual : float
        Maximum acceptable residual (in meters) for outlier detection
    
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
        'x': df['Px'].values,
        'y': df['Py'].values,
        't': t
    })
    
    n = len(data)
    half_window = window_size // 2
    
    # Arrays to store results
    keep_mask = np.zeros(n, dtype=bool)
    vx_array = np.zeros(n)
    vy_array = np.zeros(n)
    
    for i in range(half_window, n-half_window):
        if data['x'][i] < 0.02 or data['x'][i] > 1.993 - 0.02 or data['y'][i] < 0.02 or data['y'][i] > 0.992-0.02:
            continue
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
        
        try:
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
        except Exception as e:
            continue
        
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

def find_collision_points(filtered_df, wall_threshold=0.05, mallet_data=None, \
                          mallet_radius=0.1003/2, mallet_nearby_threshold=0.30, dot_prod_threshold = 0.9,\
                          mallet_time_window=0.1, wall_collisions=True, poly_degree=3):
    """
    Find collision points by detecting velocity direction changes near walls.
    
    Parameters:
    -----------
    filtered_df : pandas.DataFrame
        Filtered DataFrame with columns ['x', 'y', 't', 'vx', 'vy']
    wall_threshold : float
        Maximum distance (in meters) from wall for collision to be valid
    
    Returns:
    --------
    list of dict
        Each dict contains collision information:
        - 'point_a_idx': index of point before collision
        - 'point_b_idx': index of point after collision
        - 'collision_x': x coordinate of collision point
        - 'collision_y': y coordinate of collision point
        - 'wall': which wall ('left', 'right', 'top', 'bottom')
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

    puck_radius = 0.0629 / 2
    
    collisions = []
    
    for i in range(len(filtered_df) - 1):
        # Get adjacent points
        row_a = filtered_df.iloc[i]
        row_b = filtered_df.iloc[i + 1]
        
        # Extract velocities
        v_a = np.array([row_a['vx'], row_a['vy']])
        v_b = np.array([row_b['vx'], row_b['vy']])
        
        # Check if velocities are in different directions (dot product < 0)
        dot_product = np.dot(v_a / np.linalg.norm(v_a), v_b / np.linalg.norm(v_b))
        if dot_product >= dot_prod_threshold:
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
        
        # Determine which wall (if any) the collision is near
        wall = None
        wall_normal = None

        collision_time = (row_a['t'] + t)
        
        dist_to_left = x_int - x_min
        dist_to_right = x_max - x_int
        dist_to_bottom = y_int - y_min
        dist_to_top = y_max - y_int
        
        min_dist = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
        
        if min_dist > wall_threshold or min_dist < 0.02:
            # Not near any wall
            """
            modify the code to get the time of the colision, then check if its near a mallet.
            If so then find where the mallet would be on its path to collide with the puck
            get puck in mallet frame (puck_vel - mallet_vel) then decompose into the respective normal and tangential
            """
            if wall_collisions:
                continue

            # Not near any wall - check for mallet collision
            if mallet_data is None:
                print("No mallet data provided")
                continue  # No mallet data provided
            
            # Find mallet data points within time window around collision
            mallet_col_time = collision_time - 0.008
            time_mask = np.abs(mallet_data['t'] - mallet_col_time) <= 0.03 #mallet_time_window
            
            if np.sum(time_mask) < poly_degree + 2:
                print("Not enough points to fit polynomial")
                continue  # Not enough points to fit polynomial
            
            mallet_t = mallet_data['t'][time_mask].values
            mallet_x = mallet_data['Mx'][time_mask].values
            mallet_y = mallet_data['My'][time_mask].values

            if x_int > 1.1:
                continue
            
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
            
            #t_contact = mallet_col_time
            #mallet_x_contact = np.polyval(poly_x, t_contact)
            #mallet_y_contact = np.polyval(poly_y, t_contact)
            
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

            angle_in = np.degrees(np.arctan(abs(v_tangent_pre/v_normal_pre)))
            angle_out = np.degrees(np.arctan(-v_tangent_post/v_normal_post * (v_tangent_pre/v_normal_pre)/abs(v_tangent_pre/v_normal_pre)))
        
            
            # Calculate velocities
            v_in = np.sqrt(v_normal_pre**2 + v_tangent_pre**2)
            v_out = np.sqrt(v_normal_post**2 + v_tangent_post**2)
            
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

            angle_in = np.degrees(np.arctan(abs(v_tangent_pre/v_normal_pre)))
            angle_out = np.degrees(np.arctan(-v_tangent_post/v_normal_post * (v_tangent_pre/v_normal_pre)/abs(v_tangent_pre/v_normal_pre)))
        
            
            # Calculate velocities
            v_in = np.sqrt(v_normal_pre**2 + v_tangent_pre**2)
            v_out = np.sqrt(v_normal_post**2 + v_tangent_post**2)

            if v_out > 1.2 * v_in:
                continue

            if abs(angle_in - angle_out) > 26:
                continue
            
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
        
        if not wall_collisions:
            continue
        
        # Identify the wall and its normal vector
        if min_dist == dist_to_left:
            wall = 'left'
            wall_normal = np.array([1, 0])  # Normal points right
        elif min_dist == dist_to_right:
            wall = 'right'
            wall_normal = np.array([-1, 0])  # Normal points left
        elif min_dist == dist_to_bottom:
            wall = 'bottom'
            wall_normal = np.array([0, 1])  # Normal points up
        else:  # dist_to_top
            wall = 'top'
            wall_normal = np.array([0, -1])  # Normal points down
        
        # Calculate normal and tangential components of velocity
        # Tangent is perpendicular to normal
        wall_tangent = np.array([-wall_normal[1], wall_normal[0]])
        
        # Pre-collision velocity components (using v_a)
        v_normal_pre = np.dot(v_a, wall_normal)
        v_tangent_pre = np.dot(v_a, wall_tangent)
        
        # Post-collision velocity components (using v_b)
        v_normal_post = np.dot(v_b, wall_normal)
        v_tangent_post = np.dot(v_b, wall_tangent)
        
        # Calculate restitution coefficients
        # Normal: e_n = -v_normal_post / v_normal_pre
        # Tangential: e_t = v_tangent_post / v_tangent_pre
        
        if abs(v_normal_pre) < 1e-6:
            continue  # Avoid division by zero
        
        e_normal = -v_normal_post / v_normal_pre
        
        if abs(v_tangent_pre) < 1e-6:
            e_tangential = 1.0  # No tangential component, assume elastic
        else:
            e_tangential = v_tangent_post / v_tangent_pre

        angle_in = np.degrees(np.arctan(abs(v_tangent_pre/v_normal_pre)))
        angle_out = np.degrees(np.arctan(-v_tangent_post/v_normal_post * (v_tangent_pre/v_normal_pre)/abs(v_tangent_pre/v_normal_pre)))
    
        
        # Calculate velocities
        v_in = np.sqrt(v_normal_pre**2 + v_tangent_pre**2)
        v_out = np.sqrt(v_normal_post**2 + v_tangent_post**2)

        if v_out > 1.2 * v_in:
            continue

        if abs(angle_in - angle_out) > 26:
            continue

        if angle_in > 80:
            continue
            
        # Record collision
        collision_data = {
            'point_a_idx': i,
            'point_b_idx': i + 1,
            'collision_x': x_int,
            'collision_y': y_int,
            'wall': wall,
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
    
    return collisions

def plot_collisions(filtered_df, collisions):
    """
    Visualize detected collisions on the trajectory.
    
    Parameters:
    -----------
    filtered_df : pandas.DataFrame
        Filtered DataFrame with columns ['x', 'y']
    collisions : list of dict
        List of collision data from find_collision_points
    """
    plt.figure(figsize=(14, 7))
    
    # Plot trajectory
    plt.plot(filtered_df['x'], filtered_df['y'], 'b-', linewidth=1, alpha=0.4, label='Trajectory')
    plt.plot(filtered_df['x'], filtered_df['y'], 'bo', markersize=2, alpha=0.3)
    
    # Plot collision points
    for collision in collisions:
        x_int = collision['collision_x']
        y_int = collision['collision_y']
        plt.plot(x_int, y_int, 'r*', markersize=15, markeredgecolor='black', 
                markeredgewidth=0.5, label='Collision' if collision == collisions[0] else '')
        
        # Annotate with wall name
        plt.annotate(collision['wall'], (x_int, y_int), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Draw table boundaries
    plt.axvline(x=0, color='k', linestyle='--', linewidth=2, alpha=0.5)
    plt.axvline(x=1.9885, color='k', linestyle='--', linewidth=2, alpha=0.5)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.5)
    plt.axhline(y=0.9905, color='k', linestyle='--', linewidth=2, alpha=0.5)
    
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.title(f'Detected Collisions ({len(collisions)} found)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    # Print collision summary
    print(f"\nFound {len(collisions)} collisions:")
    for i, collision in enumerate(collisions):
        print(f"\nCollision {i+1}:")
        print(f"  Wall: {collision['wall']}")
        print(f"  Location: ({collision['collision_x']:.4f}, {collision['collision_y']:.4f})")
        print(f"  Pre-collision velocity: ({collision['vx_pre']:.3f}, {collision['vy_pre']:.3f}) m/s")
        print(f"  Post-collision velocity: ({collision['vx_post']:.3f}, {collision['vy_post']:.3f}) m/s")
        print(f"  Normal restitution (e_n): {collision['e_normal']:.3f}")
        print(f"  Tangential restitution (e_t): {collision['e_tangential']:.3f}")

def plot_collisions_mallet(filtered_df, collisions, mallet_data=None, puck_radius=0.0309, mallet_radius=0.0475, df = None):
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

    if df is not None:
        plt.plot(df['Px'], df['Py'], 'r-', linewidth=1, alpha=0.4, label='Puck Trajectory orig')
        plt.plot(df['Px'], df['Py'], 'ro', markersize=2, alpha=0.3)
    
    # Plot trajectory
    plt.plot(filtered_df['x'], filtered_df['y'], 'b-', linewidth=1, alpha=0.4, label='Puck Trajectory')
    plt.plot(filtered_df['x'], filtered_df['y'], 'bo', markersize=2, alpha=0.3)
    
    # Plot mallet trajectory if available
    if mallet_data is not None:
        plt.plot(mallet_data['Mx'], mallet_data['My'], 'b-', linewidth=1, alpha=0.3, label='Mallet Trajectory')
        plt.plot(mallet_data['Mx'], mallet_data['My'], 'bo', markersize=2, alpha=0.3)
    
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
    return n_f + (1 - n_f/n_0) * 2 / (1 + np.exp(n_r * v**2)) * n_0



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

# Example usage:
wall_collisions = True

# Load your data
#df = pd.read_csv('puck_data_2.csv')
names = ['system_loop_data.csv', 'system_loop_data2.csv', 'system_loop_data3.csv', 'system_loop_data4.csv', 'system_loop_data5.csv', 'system_loop_data6.csv']
collisions = []
for name in names:
    df = pd.read_csv(name)
    print(len(df))
    # Filter and analyze
    filtered_df = filter_and_analyze_puck_data(
        df, 
        window_size=7,           # Adjust based on your data density
        r_squared_threshold=0.97, # Adjust for stricter/looser filtering
        max_residual=0.005,
        vel_threshold=0.04        # Maximum 1cm deviation
    )

    # Plot a segment
    #plot_data_segment(filtered_df, start_idx=100, end_idx=1000)

    # Compare original vs filtered
    #plot_filtered_comparison(df, filtered_df)

    # Access the results
    if wall_collisions:
        collisions2 = find_collision_points(filtered_df, wall_threshold=0.05, mallet_data=None,\
                                            mallet_radius=None, mallet_nearby_threshold=0.30,\
                                            dot_prod_threshold = 0.9, mallet_time_window=0.1,\
                                            wall_collisions=True, poly_degree=None)
    else:
        mallet_t = np.cumsum(df['dt'].values)
        mallet_t = np.insert(mallet_t, 0, 0)[:-1]

        mallet_data = pd.DataFrame({
            'Mx': df['Mx'].values,
            'My': df['My'].values,
            'Mvx': df['Mxv'].values,
            'Mvy': df['Myv'].values,
            't': mallet_t
        })

        collisions2 = find_collision_points(filtered_df, wall_threshold=0.05, mallet_data=mallet_data,\
                                            mallet_radius=0.1011/2, mallet_nearby_threshold=0.1,\
                                            dot_prod_threshold = 0.9, mallet_time_window=0.05,\
                                            wall_collisions=False, poly_degree=3)
    collisions.extend(collisions2)

if wall_collisions:
    pass #plot_collisions(filtered_df, collisions)
else:
    plot_collisions_mallet(filtered_df, collisions, mallet_data=mallet_data, puck_radius=0.0629 / 2, mallet_radius=0.1011/2, df = df)

incoming_angles = []
outgoing_angles = []
incoming_vel = []
outgoing_vel = []

for collision in collisions:
    # Calculate incoming angle
    v_normal_pre = collision['v_normal_pre']
    v_tangent_pre = collision['v_tangent_pre']
    angle_in = np.degrees(np.arctan(abs(v_tangent_pre/v_normal_pre)))
    
    # Calculate outgoing angle
    v_normal_post = collision['v_normal_post']
    v_tangent_post = collision['v_tangent_post']
    angle_out = np.degrees(np.arctan(-v_tangent_post/v_normal_post * (v_tangent_pre/v_normal_pre)/abs(v_tangent_pre/v_normal_pre)))
    
    # Calculate velocities
    v_in = np.sqrt(v_normal_pre**2 + v_tangent_pre**2)
    v_out = np.sqrt(v_normal_post**2 + v_tangent_post**2)
    
    incoming_angles.append(angle_in)
    outgoing_angles.append(angle_out)
    incoming_vel.append(v_in)
    outgoing_vel.append(v_out)


for i in range(6):
    incoming_angles.append(15*i+10)
    outgoing_angles.append(15*i+10+2.5*(5-i))
    incoming_vel.append(14)
    outgoing_vel.append(6.5+0.8+(10.5-6.5-0.8)*i/5) #10.5

for i in range(6):
    incoming_angles.append(15*i+10)
    outgoing_angles.append(15*i+10+2.5*(5-i))
    incoming_vel.append(11.5)
    outgoing_vel.append(6.1+3.1/5+(9.2-6.1-3.1/5)*i/5) #10.5


# Convert to numpy arrays
incoming_angles = np.array(incoming_angles)
outgoing_angles = np.array(outgoing_angles)
incoming_vel = np.array(incoming_vel)
outgoing_vel = np.array(outgoing_vel)

# Prepare data for neural network
X = np.column_stack([incoming_angles, incoming_vel])  # Inputs: angle_in, vel_in
y = np.column_stack([outgoing_angles, outgoing_vel])  # Outputs: angle_out, vel_out

# Split data
X_train = X
y_train = y

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=42)

# Scale the data
class HeteroscedasticNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=8, output_dim=2):
        super(HeteroscedasticNN, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.predictions = nn.Linear(hidden_dim, output_dim)
        self.log_sigmas = nn.Linear(hidden_dim, output_dim)
        
        # Scale parameter for sigmas (initialized to 1.0, set later for denormalization)
        self.register_buffer('sigma_scale', torch.ones(output_dim))
        
    def forward(self, x):
        x = F.softplus(self.fc1(x))
        x = F.softplus(self.fc2(x))
        
        # Predictions
        preds = self.predictions(x)
        
        # Sigmas: softplus to ensure positive, add epsilon, then scale
        log_sig = self.log_sigmas(x)
        sigmas = (F.softplus(log_sig) + 1e-6) * self.sigma_scale
        
        # Concatenate predictions and sigmas
        return torch.cat([preds, sigmas], dim=1)

# Gaussian negative log-likelihood loss
def gaussian_nll_loss(y_pred, y_true):
    """
    y_pred: [batch_size, 4] containing [pred1, pred2, sigma1, sigma2]
    y_true: [batch_size, 2] containing [true1, true2]
    """
    predictions = y_pred[:, :2]
    sigmas = y_pred[:, 2:]
    
    squared_error = (y_true - predictions) ** 2
    variance = sigmas ** 2
    
    nll = 0.5 * torch.log(variance) + squared_error / (2 * variance)
    
    return nll.mean()

# Training setup
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled)

model_path = 'collision_model_heteroscedastic.pt' if wall_collisions else 'mallet_collision_model_heteroscedastic.pt'

"""
# Create model
model = HeteroscedasticNN(input_dim=2, hidden_dim=8, output_dim=2)
optimizer = torch.optim.Adam(model.parameters())

# Training loop
print("Training heteroscedastic neural network...")

model.train()
for epoch in range(20000):
    optimizer.zero_grad()
    
    # Forward pass
    y_pred = model(X_train_tensor)
    loss = gaussian_nll_loss(y_pred, y_train_tensor)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/3000, Loss: {loss.item():.6f}")

print("Training complete!")

# Embed normalization into model weights
def embed_normalization_into_model(model, scaler_X, scaler_y):
    model.eval()
    
    X_mean = torch.FloatTensor(scaler_X.mean_)
    X_std = torch.FloatTensor(scaler_X.scale_)
    y_mean = torch.FloatTensor(scaler_y.mean_)
    y_std = torch.FloatTensor(scaler_y.scale_)
    
    # === Embed input normalization into first layer ===
    with torch.no_grad():
        W1 = model.fc1.weight.data  # Shape: [hidden_dim, input_dim]
        b1 = model.fc1.bias.data    # Shape: [hidden_dim]
        
        # PyTorch uses transposed weight convention: output = x @ W^T + b
        # For normalization: x_norm = (x - mean) / std
        # New: output = ((x - mean) / std) @ W^T + b
        #            = x @ (W^T / std) - (mean / std) @ W^T + b
        W1_new = W1 / X_std.unsqueeze(0)  # Divide each column by std
        b1_new = b1 - (X_mean / X_std) @ W1.T
        
        model.fc1.weight.data = W1_new
        model.fc1.bias.data = b1_new
        
        # === Embed output denormalization into predictions layer ===
        W_pred = model.predictions.weight.data
        b_pred = model.predictions.bias.data
        
        # Transform: y_denorm = y_norm * std + mean
        W_pred_new = W_pred * y_std.unsqueeze(1)
        b_pred_new = b_pred * y_std + y_mean
        
        model.predictions.weight.data = W_pred_new
        model.predictions.bias.data = b_pred_new
        
        # === Set sigma scaling ===
        model.sigma_scale.copy_(y_std)
    
    return model

# Embed normalization
embed_normalization_into_model(model, scaler_X, scaler_y)

# Save model

torch.save(model.state_dict(), model_path)
#print(f"Model saved to {model_path}")

# To save the entire model (including architecture):
#torch.save(model, model_path.replace('.pt', '_full.pt'))

# Make predictions (no scaling needed!)
model.eval()

# ===== LOADING THE MODEL =====
"""

# Option 1: Load state dict (requires model definition)
# Option 2: Load full model (if saved with torch.save(model, ...))
model = HeteroscedasticNN(input_dim=2, hidden_dim=8, output_dim=2)
model.load_state_dict(torch.load(model_path))
model.eval()


# Use loaded model
with torch.no_grad():
    y_pred_full = model(torch.FloatTensor(X)).numpy()


y_pred = y_pred_full[:,:2]

angle_pred = y_pred[:, 0]
vel_pred = y_pred[:, 1]

sigma_angle_pred = y_pred_full[:,2]
sigma_vel_pred = y_pred_full[:,3]

# Calculate residuals
residuals_angle = outgoing_angles - angle_pred
residuals_vel = outgoing_vel - vel_pred
sigma_angle = np.std(residuals_angle)
sigma_vel = np.std(residuals_vel)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

scatter1 = ax1.scatter(incoming_angles, sigma_angle_pred, c=incoming_vel, 
                       cmap='viridis', s=50, alpha=0.6)

angle_range = np.linspace(0, 90, 200)
vel_values = np.arange(0.01, 15.1, 1.0)  # 0.01 to 15 m/s, step ~1 m/s

colors = plt.cm.viridis(np.linspace(0, 1, len(vel_values)))

for i, vel_fixed in enumerate(vel_values):
    # Create input array
    X_pred = np.column_stack([angle_range, np.full_like(angle_range, vel_fixed)])
    
    # Predict
    with torch.no_grad():
        y_pred_full = model(torch.FloatTensor(X_pred)).numpy()
    angle_out_sigma = y_pred_full[:,2]
    
    ax1.plot(angle_range, angle_out_sigma, linewidth=2, color=colors[i], 
             label=f'v_in = {vel_fixed:.1f} m/s')
    
ax1.set_xlabel('Incoming Angle (degrees)', fontsize=12)
ax1.set_ylabel('Predicted σ_angle (degrees)', fontsize=12)
ax1.set_title('Predicted Angle Uncertainty vs Input Angle', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=ax1, label='Incoming Velocity (m/s)')

vel_range = np.linspace(0.01, 15, 200)
angle_values = np.arange(0, 91, 10)  # 0 to 90 degrees, step 10

colors2 = plt.cm.plasma(np.linspace(0, 1, len(angle_values)))

for i, angle_fixed in enumerate(angle_values):
    # Create input array
    X_pred = np.column_stack([np.full_like(vel_range, angle_fixed), vel_range])
    
    # Predict
    with torch.no_grad():
        y_pred_full = model(torch.FloatTensor(X_pred)).numpy()
    vel_out_sigma = y_pred_full[:,3]
    
    ax2.plot(vel_range, vel_out_sigma, linewidth=2, color=colors2[i], 
             label=f'θ_in = {angle_fixed:.0f}°')

scatter2 = ax2.scatter(incoming_vel, sigma_vel_pred, c=incoming_angles, 
                       cmap='plasma', s=50, alpha=0.6)
ax2.set_xlabel('Incoming Velocity (m/s)', fontsize=12)
ax2.set_ylabel('Predicted σ_vel (m/s)', fontsize=12)
ax2.set_title('Predicted Velocity Uncertainty vs Input Velocity', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label='Incoming Angle (degrees)')

plt.tight_layout()
plt.show()

# Create plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))

# ============== PLOT 1: Predicted vs Actual Angles ==============
scatter1 = ax1.scatter(outgoing_angles, angle_pred, 
                       c=incoming_vel, cmap='viridis', 
                       s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
lim_angle = [min(outgoing_angles.min(), angle_pred.min()), 
             max(outgoing_angles.max(), angle_pred.max())]
ax1.plot(lim_angle, lim_angle, 'r--', linewidth=2, label='Perfect prediction')
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('Incoming Velocity (m/s)', fontsize=11)
ax1.set_xlabel('Actual Outgoing Angle (degrees)', fontsize=12)
ax1.set_ylabel('Predicted Outgoing Angle (degrees)', fontsize=12)
ax1.set_title(f'NN Angle Prediction (σ = {sigma_angle:.2f}°)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# ============== PLOT 2: Predicted vs Actual Velocities ==============
scatter2 = ax2.scatter(outgoing_vel, vel_pred, 
                       c=incoming_angles, cmap='plasma', 
                       s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
lim_vel = [min(outgoing_vel.min(), vel_pred.min()), 
           max(outgoing_vel.max(), vel_pred.max())]
ax2.plot(lim_vel, lim_vel, 'r--', linewidth=2, label='Perfect prediction')
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('Incoming Angle (degrees)', fontsize=11)
ax2.set_xlabel('Actual Outgoing Velocity (m/s)', fontsize=12)
ax2.set_ylabel('Predicted Outgoing Velocity (m/s)', fontsize=12)
ax2.set_title(f'NN Velocity Prediction (σ = {sigma_vel:.4f} m/s)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# ============== PLOT 3: Angle Residuals ==============
ax3.scatter(incoming_angles, residuals_angle, c=incoming_vel, 
            cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax3.axhline(y=sigma_angle, color='orange', linestyle=':', linewidth=1.5, label=f'±1σ = ±{sigma_angle:.2f}°')
ax3.axhline(y=-sigma_angle, color='orange', linestyle=':', linewidth=1.5)
ax3.set_xlabel('Incoming Angle (degrees)', fontsize=12)
ax3.set_ylabel('Residual (Actual - Predicted) Angle (degrees)', fontsize=12)
ax3.set_title('Angle Prediction Residuals', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# ============== PLOT 4: Velocity Residuals ==============
ax4.scatter(incoming_vel, residuals_vel, c=incoming_angles, 
            cmap='plasma', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax4.axhline(y=sigma_vel, color='orange', linestyle=':', linewidth=1.5, label=f'±1σ = ±{sigma_vel:.4f} m/s')
ax4.axhline(y=-sigma_vel, color='orange', linestyle=':', linewidth=1.5)
ax4.set_xlabel('Incoming Velocity (m/s)', fontsize=12)
ax4.set_ylabel('Residual (Actual - Predicted) Velocity (m/s)', fontsize=12)
ax4.set_title('Velocity Prediction Residuals', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.show()

fig3, ax5 = plt.subplots(figsize=(12, 8))

angle_range = np.linspace(0, 90, 200)
vel_values = np.arange(0.01, 8.1, 1.0)  # 0.01 to 15 m/s, step ~1 m/s

colors = plt.cm.viridis(np.linspace(0, 1, len(vel_values)))

for i, vel_fixed in enumerate(vel_values):
    # Create input array
    X_pred = np.column_stack([angle_range, np.full_like(angle_range, vel_fixed)])
    
    # Predict
    with torch.no_grad():
        y_pred_full = model(torch.FloatTensor(X_pred)).numpy()
    y_pred = y_pred_full[:,:2]
    angle_out_pred = y_pred[:, 0]
    
    ax5.plot(angle_range, angle_out_pred, linewidth=2, color=colors[i], 
             label=f'v_in = {vel_fixed:.1f} m/s')

ax5.scatter(incoming_angles, outgoing_angles, c=incoming_vel, 
            cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax5.plot([0, 90], [0, 90], 'k--', linewidth=1.5, alpha=0.3, label='y = x reference')
ax5.set_xlabel('Input Angle (degrees)', fontsize=12)
ax5.set_ylabel('Output Angle (degrees)', fontsize=12)
ax5.set_title('NN Prediction: Output Angle vs Input Angle\n(for different input velocities)', 
              fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax5.set_xlim([0, 90])
plt.tight_layout()
plt.show()

# Plot 2: Output vel vs Input vel for different angles
fig4, ax6 = plt.subplots(figsize=(12, 8))

vel_range = np.linspace(0.01, 15, 200)
angle_values = np.arange(0, 91, 10)  # 0 to 90 degrees, step 10

colors2 = plt.cm.plasma(np.linspace(0, 1, len(angle_values)))

for i, angle_fixed in enumerate(angle_values):
    # Create input array
    X_pred = np.column_stack([np.full_like(vel_range, angle_fixed), vel_range])
    
    # Predict
    with torch.no_grad():
        y_pred_full = model(torch.FloatTensor(X_pred)).numpy()
    y_pred = y_pred_full[:,:2]
    vel_out_pred = y_pred[:, 1]
    
    ax6.plot(vel_range, vel_out_pred, linewidth=2, color=colors2[i], 
             label=f'θ_in = {angle_fixed:.0f}°')

ax6.scatter(incoming_vel, outgoing_vel, c=incoming_angles, 
            cmap='plasma', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax6.plot([0, 15], [0, 15], 'k--', linewidth=1.5, alpha=0.3, label='y = x reference')
ax6.set_xlabel('Input Velocity (m/s)', fontsize=12)
ax6.set_ylabel('Output Velocity (m/s)', fontsize=12)
ax6.set_title('NN Prediction: Output Velocity vs Input Velocity\n(for different input angles)', 
              fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax6.set_xlim([0, 15])
plt.tight_layout()
plt.show()

# Print statistics
print("\n" + "="*60)
print("NEURAL NETWORK STATISTICS")
print("="*60)
print(f"Number of collisions: {len(collisions)}")
print(f"Training samples: {len(X_train)}")
print()
print("ANGLE PREDICTIONS:")
print(f"  Residual std deviation (σ): {sigma_angle:.3f}°")
print(f"  Mean absolute error: {np.mean(np.abs(residuals_angle)):.3f}°")
print(f"  Max absolute error: {np.max(np.abs(residuals_angle)):.3f}°")
print()
print("VELOCITY PREDICTIONS:")
print(f"  Residual std deviation (σ): {sigma_vel:.4f} m/s")
print(f"  Mean absolute error: {np.mean(np.abs(residuals_vel)):.4f} m/s")
print(f"  Max absolute error: {np.max(np.abs(residuals_vel)):.4f} m/s")
print(f"  Relative error (σ/mean_v_out): {sigma_vel/np.mean(outgoing_vel)*100:.2f}%")

# Plot training history
fig2, (ax_loss, ax_mae) = plt.subplots(1, 2, figsize=(14, 5))

ax_loss.plot(history.history['loss'], label='Training Loss')
ax_loss.set_xlabel('Epoch', fontsize=12)
ax_loss.set_ylabel('Loss (MSE)', fontsize=12)
ax_loss.set_title('Training History - Loss', fontsize=13, fontweight='bold')
ax_loss.legend()
ax_loss.grid(True, alpha=0.3)
"""
ax_mae.plot(history.history['mae'], label='Training MAE')
ax_mae.set_xlabel('Epoch', fontsize=12)
ax_mae.set_ylabel('Mean Absolute Error', fontsize=12)
ax_mae.set_title('Training History - MAE', fontsize=13, fontweight='bold')
ax_mae.legend()
ax_mae.grid(True, alpha=0.3)
"""
plt.tight_layout()
plt.show()


#POLYNOMIAL FIT
"""
incoming_angles = []
outgoing_angles = []
total_velocities = []
outgoing_vel = []

for collision in collisions:
    # Calculate incoming angle (angle between incoming velocity and normal)
    v_normal_pre = collision['v_normal_pre']
    v_tangent_pre = collision['v_tangent_pre']
    angle_in = np.degrees(np.arctan(abs(v_tangent_pre/v_normal_pre)))
    
    # Calculate outgoing angle (angle between outgoing velocity and normal)
    v_normal_post = collision['v_normal_post']
    v_tangent_post = collision['v_tangent_post']
    angle_out = np.degrees(np.arctan(abs(v_tangent_post/v_normal_post)))
    
    # Calculate total velocity magnitude
    vx_pre = collision['vx_pre']
    vy_pre = collision['vy_pre']
    v_total = np.sqrt(vx_pre**2 + vy_pre**2)

    v_out = np.sqrt(collision['v_normal_post']**2 + collision['v_tangent_post']**2)
    
    incoming_angles.append(angle_in)
    outgoing_angles.append(angle_out)
    total_velocities.append(v_total)
    outgoing_vel.append(v_out)

# Convert to numpy arrays
incoming_angles = np.array(incoming_angles)
outgoing_angles = np.array(outgoing_angles)
total_velocities = np.array(total_velocities)
outgoing_vel = np.array(outgoing_vel)

# ============== FIRST PLOT: ANGLES ==============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Sort data for fitting
sort_idx = np.argsort(incoming_angles)
x_angle = incoming_angles[sort_idx]
y_angle = outgoing_angles[sort_idx]

# Get corresponding velocities for the angle data
incoming_vel_for_angles = []
for collision in collisions:
    v_normal_pre = collision['v_normal_pre']
    v_tangent_pre = collision['v_tangent_pre']
    v_in = np.sqrt(v_normal_pre**2 + v_tangent_pre**2)
    incoming_vel_for_angles.append(v_in)
incoming_vel_for_angles = np.array(incoming_vel_for_angles)

# Piecewise fit function for angles with velocity dependence
def angle_fit_func(X, a, b, c, d, e, f):
    
    #Piecewise function with velocity dependence:
    #- 0 to 30 deg: theta^2 / a(v)  where a(v) = a * (1 + v/v_scale)
    #- 30 to 90 deg: b*theta * sigmoid((theta-c)/d) + E
    #where E ensures continuity at theta=30
    
    #X is a 2D array where X[0] = theta, X[1] = velocity
    
    result = a + b*X[0]+c*X[1] + d*X[0]**2 + e*X[1]**2 + f*X[0]*X[1]

    return result

# Prepare data for 2D fitting
X_data = np.vstack([incoming_angles, incoming_vel_for_angles])

# Initial guess
p0_angle = [45, 0, 0, 0, 0, 0]
popt_angle, _ = curve_fit(angle_fit_func, X_data, outgoing_angles, p0=p0_angle, maxfev=10000)

# Calculate residuals and std
y_angle_pred = angle_fit_func(X_data, *popt_angle)
residuals_angle = outgoing_angles - y_angle_pred
sigma_angle = np.std(residuals_angle)

# Calculate E for display (using mean velocity)
a, b, c, d, e, f = popt_angle

fit_label_angle = f'Piecewise fit with velocity (σ={sigma_angle:.2f}°)\na={a:.1f}, b={b:.2f}, c={c:.1f}'

# Plot
scatter1 = ax1.scatter(incoming_angles, outgoing_angles, 
                       c=total_velocities, cmap='binary', 
                       s=50, alpha=0.5, edgecolors='black', linewidth=0.5)

# Generate fit curves for the same velocity range as the NN plot
x_angle_fit = np.linspace(0, 90, 500)
vel_values = np.arange(0.01, 8.1, 1.0)  # Same as NN plot
colors = plt.cm.viridis(np.linspace(0, 1, len(vel_values)))

for i, vel_fixed in enumerate(vel_values):
    X_fit = np.vstack([x_angle_fit, np.full_like(x_angle_fit, vel_fixed)])
    y_angle_fit = angle_fit_func(X_fit, *popt_angle)
    ax1.plot(x_angle_fit, y_angle_fit, linewidth=2, color=colors[i], alpha=0.7,
             label=f'v={vel_fixed:.1f} m/s' if i % 3 == 0 else '')  # Label every 3rd to avoid clutter

ax1.plot(x_angle_fit, x_angle_fit, 'k--', alpha=0.3, linewidth=1, label='y = x reference')

cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('Total Velocity (m/s)', fontsize=11)

ax1.set_xlabel('Incoming Angle (degrees)', fontsize=12)
ax1.set_ylabel('Outgoing Angle (degrees)', fontsize=12)
ax1.set_title(f'Collision Angles: Incoming vs Outgoing\nParametric Fit (σ={sigma_angle:.2f}°)', 
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8, loc='best')

# ============== SECOND PLOT: VELOCITIES ==============
# Initial guess

p0_vel = [3, 0, 0, 0, 0, 0]
popt_vel, _ = curve_fit(angle_fit_func, X_data, outgoing_vel, p0=p0_vel, maxfev=10000)

# Calculate residuals and std
y_vel_pred = angle_fit_func(X_data, *popt_vel)
residuals_vel = outgoing_vel - y_vel_pred
sigma_vel = np.std(residuals_vel)

# Calculate E for display (using mean velocity)
a, b, c, d, e, f = popt_vel

# Plot
scatter2 = ax2.scatter(total_velocities, outgoing_vel, 
                       c=incoming_angles, cmap='binary', 
                       s=50, alpha=0.5, edgecolors='black', linewidth=0.5)

# Generate fit curves for the same velocity range as the NN plot
x_vel_fit = np.linspace(0, 8, 500)
angle_vals = np.arange(0, 90, 10)  # Same as NN plot
colors = plt.cm.viridis(np.linspace(0, 1, len(angle_vals)))

for i, angle_fixed in enumerate(angle_vals):
    X_fit = np.vstack([np.full_like(x_vel_fit, angle_fixed), x_vel_fit])
    y_vel_fit = angle_fit_func(X_fit, *popt_vel)
    ax2.plot(x_vel_fit, y_vel_fit, linewidth=2, color=colors[i], alpha=0.7,
             label=f'angle={angle_fixed:.1f} degrees' if i % 3 == 0 else '')  # Label every 3rd to avoid clutter

ax2.plot(x_vel_fit, x_vel_fit, 'k--', alpha=0.3, linewidth=1, label='y = x reference')

cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('Angle (degrees)', fontsize=11)

ax2.set_xlabel('Incoming Vel (m/s)', fontsize=12)
ax2.set_ylabel('Outgoing Vel (m/s)', fontsize=12)
ax2.set_title(f'Collision Vel: Incoming vs Outgoing\nParametric Fit (σ={sigma_vel:.2f} m/s)', 
              fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8, loc='best')

plt.show()
"""


# Print statistics
"""
print("="*60)
print("ANGLE FIT STATISTICS")
print("="*60)
print(f"Number of collisions: {len(collisions)}")
print(f"Residual standard deviation (σ): {sigma_angle:.3f}°")
print(f"Mean absolute error: {np.mean(np.abs(residuals_angle)):.3f}°")
print(f"Max absolute error: {np.max(np.abs(residuals_angle)):.3f}°")
print()
print("="*60)
print("VELOCITY FIT STATISTICS")
print("="*60)
print(f"Residual standard deviation (σ): {sigma_vel:.4f} m/s")
print(f"Mean absolute error: {np.mean(np.abs(residuals_vel)):.4f} m/s")
print(f"Max absolute error: {np.max(np.abs(residuals_vel)):.4f} m/s")
print(f"Relative error (σ/mean_v_out): {sigma_vel/np.mean(outgoing_vel)*100:.2f}%")
"""

# Visualize collisions


#fit_results = fit_restitution_model(collisions)

# Plot restitution analysis
#plot_restitution_analysis(collisions, fit_results)

# Access fitted parameters
#print(f"Normal params: {fit_results['normal_params']}")
#print(f"Tangential params: {fit_results['tangential_params']}")
#print(f"Sigma normal: {fit_results['sigma_normal']}")
#print(f"Sigma tangent: {fit_results['sigma_tangent']}")
