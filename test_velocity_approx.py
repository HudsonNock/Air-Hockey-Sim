import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Load CSV ---
df = pd.read_csv("feedforward_data_12V.csv")

pos_err = []
vel_err = []

# --- Choose a segment (say first 200 rows for demo) ---
for i in range(10, 100):
    print(i)
    segment = df.iloc[i*100+75:i*100+100+20].reset_index(drop=True)

    # --- Parameters ---
    window_size = 11   # must be odd for regression center
    half_window = window_size // 2

    # --- Extract arrays ---
    x = segment["x"].to_numpy()
    y = segment["y"].to_numpy()
    dt = segment["dt"].to_numpy()

    #print(dt)

    # --- Cumulative time (needed for regression) ---
    t = np.cumsum(dt)

    t = t[::2]
    x = x[::2]
    y = y[::2]

        # ======================
    # 2) Causal Linear regression method
    # ======================
    # Fit x(t), y(t) linearly
    n = half_window
    t_half = t[:window_size] / 1000.0
    x_half = x[:window_size]
    y_half = y[:window_size]
    causal_coef_x = np.polyfit(t_half, x_half, 2)
    causal_coef_y = np.polyfit(t_half, y_half, 2)

    center_idx = half_window
    t_center_causal = t_half[-1]
    pos_center = np.array([x_half[-1], y_half[-1]])

    causal_ax_reg, causal_ay_reg = causal_coef_x[0]*2, causal_coef_y[0]*2  # slopes = velocity components
    causal_vx_reg, causal_vy_reg = causal_coef_x[0]*2*t_center_causal + causal_coef_x[1], causal_coef_y[0]*2*t_center_causal + causal_coef_y[1]
    causal_mag_reg = np.sqrt(causal_vx_reg**2 + causal_vy_reg**2)
    causal_dir_reg = np.arctan2(causal_vy_reg, causal_vx_reg)

    causal_a_mag_reg = np.sqrt(causal_ax_reg**2 + causal_ay_reg**2)
    causal_a_dir_reg = np.arctan2(causal_ay_reg, causal_ax_reg)

        # --- Pick the first window in the segment ---
        # 
    # ======================
    # 1) Linear regression method
    # ======================
    # Fit x(t), y(t) linearly
    idx = None
    for i in range(len(t)):
        if t[i]/1000.0 >= t_half[-1]:
            idx = i
            break

    t_win = (t[idx-half_window:idx-half_window+window_size] / 1000)
    x_win = x[idx-half_window:idx-half_window+window_size]
    y_win = y[idx-half_window:idx-half_window+window_size]

    coef_x = np.polyfit(t_win, x_win, 2)
    coef_y = np.polyfit(t_win, y_win, 2)

    center_idx = half_window
    t_center = t_half[-1]
    pos_center = np.array([x_win[center_idx], y_win[center_idx]])

    ax_reg, ay_reg = coef_x[0]*2, coef_y[0]*2  # slopes = velocity components
    vx_reg, vy_reg = coef_x[0]*2*t_center + coef_x[1], coef_y[0]*2*t_center + coef_y[1]
    mag_reg = np.sqrt(vx_reg**2 + vy_reg**2)
    dir_reg = np.arctan2(vy_reg, vx_reg)

    a_mag_reg = np.sqrt(ax_reg**2 + ay_reg**2)
    a_dir_reg = np.arctan2(ay_reg, ax_reg)

    # Position of vector: center of window

    # ======================
    # 2) Weighted linear-fit (causal, left-half only)
    # ======================
    # We'll use same length window but only use left-half (past values up to center)

    # Precompute weights for equally spaced samples (linear regression slope)
    # slope = sum((ti - mean_t) * (xi - mean_x)) / sum((ti - mean_t)^2)
    def regression_slope(t_vals, v_vals):
        t_mean = np.mean(t_vals)
        v_mean = np.mean(v_vals)
        return np.sum((t_vals - t_mean)*(v_vals - v_mean)) / np.sum((t_vals - t_mean)**2)

    vx_wlf = regression_slope(t_half, x_half)
    vy_wlf = regression_slope(t_half, y_half)

    mag_wlf = np.sqrt(vx_wlf**2 + vy_wlf**2)
    dir_wlf = np.arctan2(vy_wlf, vx_wlf)

    # Position of vector: at the last point of the half window
    pos_half = np.array([x_half[-1], y_half[-1]])

    # ======================
    # Print results
    # ======================
    
    """
    print("Linear Regression Vel (symmetric window):")
    print(f"  Magnitude = {mag_reg:.3f}, Direction = {np.degrees(dir_reg):.2f} deg")

    print("Linear Regression Acc (symmetric window):")
    print(f"  Magnitude = {a_mag_reg:.3f}, Direction = {np.degrees(a_dir_reg):.2f} deg")

    print("Causal Linear Regression Vel (symmetric window):")
    print(f"  Magnitude = {causal_mag_reg:.3f}, Direction = {np.degrees(causal_dir_reg):.2f} deg")

    print("Causal Linear Regression Acc (symmetric window):")
    print(f"  Magnitude = {causal_a_mag_reg:.3f}, Direction = {np.degrees(causal_a_dir_reg):.2f} deg")
    
    # ======================
    # Plot
    # ======================
    plt.figure(figsize=(6,6))
    plt.scatter(x, y, s=5, label="x Trajectory")

    t_fit = np.linspace(t_win[0], t_win[-1], 100)

    # Evaluate polynomial at each t
    x_fit = np.polyval(coef_x, t_fit)
    y_fit = np.polyval(coef_y, t_fit)
    plt.plot(x_fit, y_fit, color="green", label="quadratic regression")

    t_fit = np.linspace(t_half[0], t_half[-1], 60)

    causal_x_fit = np.polyval(causal_coef_x, t_fit)
    causal_y_fit = np.polyval(causal_coef_y, t_fit)

    # Plot scatter of data in the window
    #plt.scatter(x_win, y_win, color="gray", s=10, label="window data")

    # Plot the quadratic regression curve
    plt.plot(causal_x_fit, causal_y_fit, color="orange", label="causal quadratic regression")

    t_point = t_half[-1]
    x_fit = np.polyval(coef_x, t_point)
    y_fit = np.polyval(coef_y, t_point)
    
    # Arrow for regression
    plt.arrow(x_fit, y_fit, vx_reg/mag_reg*0.001, vy_reg/mag_reg*0.001, 
            color="red", width=0.0001, head_width=0.001, label="Regression Vel")

    # Arrow for weighted linear-fit
    #plt.arrow(pos_half[0], pos_half[1], vx_wlf/mag_wlf*0.001, vy_wlf/mag_wlf*0.001, 
    #        color="blue", width=0.0001, head_width=0.001, label="Weighted LF")
    
    plt.arrow(x_fit, y_fit, ax_reg/a_mag_reg*0.001, ay_reg/a_mag_reg*0.001, 
            color="green", width=0.0001, head_width=0.001, label="Regression Acc")
    
    pos_err.append([x_fit - causal_x_fit[-1], y_fit - causal_y_fit[-1]])
    vel_err.append([vx_reg - causal_vx_reg, vy_reg - causal_vy_reg])
    
    plt.arrow(causal_x_fit[-1], causal_y_fit[-1], causal_ax_reg/causal_a_mag_reg*0.001, causal_ay_reg/causal_a_mag_reg*0.001, 
            color="blue", width=0.0001, head_width=0.001, label="Causal Regression Acc")
    
    plt.arrow(causal_x_fit[-1], causal_y_fit[-1], causal_vx_reg/causal_mag_reg*0.001, causal_vy_reg/causal_mag_reg*0.001, 
            color="yellow", width=0.0001, head_width=0.001, label="Causal Regression Vel")
    
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.title("Velocity Estimation Methods")
    plt.show(block=True)
    """

    t_fit = np.linspace(t_win[0], t_win[-1], 100)

    x_fit = np.polyval(coef_x, t_fit)
    y_fit = np.polyval(coef_y, t_fit)

    t_fit = np.linspace(t_half[0], t_half[-1], 60)

    causal_x_fit = np.polyval(causal_coef_x, t_fit)
    causal_y_fit = np.polyval(causal_coef_y, t_fit)

    t_point = t_half[-1]
    x_fit = np.polyval(coef_x, t_point)
    y_fit = np.polyval(coef_y, t_point)

    pos_err.append([x_fit - causal_x_fit[-1], y_fit - causal_y_fit[-1]])
    vel_err.append([vx_reg - causal_vx_reg, vy_reg - causal_vy_reg])

pos_err = np.array(pos_err)
vel_err = np.array(vel_err)
print(f"Pos x std: {np.std(pos_err[:,0])}")
print(f"Pos y std: {np.std(pos_err[:,1])}")
print(f"Vel x std: {np.std(vel_err[:,0])}")
print(f"Vel y std: {np.std(vel_err[:,1])}")