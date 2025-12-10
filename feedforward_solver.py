import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('mallet_data_newp_supercap2.csv')
df2 = pd.read_csv('mallet_data_oldp_supercap2.csv')
df3 = pd.read_csv('mallet_data_overhead_supercap2.csv')
#read up until zero entries
df = df[:27825]
df2 = df2[:25573]
df3 = df3[:23365]
pullyR = (df['x'][0] + df2['x'][0] + df3['x'][0])/3


# Remove the second row (index 1) which contains zeros
df = df.drop(index=1).reset_index(drop=True)
df2 = df2.drop(index=1).reset_index(drop=True)
df3 = df3.drop(index=1).reset_index(drop=True)

dfs = [df, df2, df3]

Vxs = None
Vys = None
Axs = None
Ays = None

for j in range(3):
    # Extract data
    x = dfs[j]['x'].values
    y = dfs[j]['y'].values
    left_pwm = dfs[j]['Left_PWM'].values
    right_pwm = dfs[j]['Right_PWM'].values
    dt = dfs[j]['dt'].values / 1000.0

    # Calculate Vx
    Vx = (left_pwm - right_pwm)* 24
    Vy = (-left_pwm - right_pwm) * 24

    # Calculate time array (cumulative sum of dt)
    t = np.concatenate([[0], np.cumsum(dt[:-1])])

    # Function to compute derivatives using Savitzky-Golay filter
    def compute_derivatives(data, t_data, window=21, poly_order=4):
        """
        Compute first, second, and third derivatives using sliding window polynomial fits
        """
        n = len(data)
        half_window = window // 2
        
        # Initialize derivative arrays
        data_dot = np.zeros(n)
        data_ddot = np.zeros(n)
        data_dddot = np.zeros(n)
        
        for i in range(n):
            # Define window bounds
            start_idx = max(0, i - half_window)
            end_idx = min(n, i + half_window + 1)
            
            # Extract window data
            t_window = t_data[start_idx:end_idx]
            data_window = data[start_idx:end_idx]
            
            # Fit polynomial
            coeffs = np.polyfit(t_window, data_window, poly_order)
            
            # Compute derivative polynomials
            # p(t) = c0*t^n + c1*t^(n-1) + ... + cn
            # p'(t) = n*c0*t^(n-1) + (n-1)*c1*t^(n-2) + ...
            deriv1_coeffs = np.polyder(coeffs, 1)
            deriv2_coeffs = np.polyder(coeffs, 2)
            deriv3_coeffs = np.polyder(coeffs, 3)
            
            # Evaluate at center time point
            t_center = t_data[i]
            data_dot[i] = np.polyval(deriv1_coeffs, t_center)
            data_ddot[i] = np.polyval(deriv2_coeffs, t_center)
            data_dddot[i] = np.polyval(deriv3_coeffs, t_center)
        
        return data_dot, data_ddot, data_dddot

    # Compute derivatives
    window_y = 35
    window_x = 91

    x_dot, x_ddot, x_dddot = compute_derivatives(x, t, window=window_x, poly_order=3)
    y_dot, y_ddot, y_dddot = compute_derivatives(y, t, window=window_y, poly_order=3)

    x_dot = x_dot[window_x:len(Vx)-window_x]
    x_ddot = x_ddot[window_x:len(Vx)-window_x]
    x_dddot = x_dddot[window_x:len(Vx)-window_x]
    tx = t[window_x:len(Vx)-window_x]
    Vx = Vx[window_x:len(Vx)-window_x]

    y_dot = y_dot[window_y:len(Vy)-window_y]
    y_ddot = y_ddot[window_y:len(Vy)-window_y]
    y_dddot = y_dddot[window_y:len(Vy)-window_y]
    ty = t[window_y:len(Vy)-window_y]
    Vy = Vy[window_y:len(Vy)-window_y]

    Ax = np.column_stack([x_dddot, x_ddot, x_dot])
    Ay = np.column_stack([y_dddot, y_ddot, y_dot])

    if j == 0:
        Vxs = Vx
        Vys = Vy
        Axs = Ax
        Ays = Ay
    else:
        Vxs = np.concatenate([Vxs, Vx])
        Vys = np.concatenate([Vys, Vy])
        Axs = np.concatenate([Axs, Ax], axis=0)
        Ays = np.concatenate([Ays, Ay], axis=0)



# Set up the least squares problem
# We want to solve: Vx = a*x''' + b*x'' + c*x'
# This is a linear regression problem: Vx = [x''', x'', x'] * [a, b, c]^T

# Create the design matrix


# Solve using least squares
coeffsx, residualsx, rankx, sx = np.linalg.lstsq(Ax, Vx, rcond=None)
coeffsy, residualsy, ranky, sy = np.linalg.lstsq(Ay, Vy, rcond=None)

a_x, b_x, c_x = coeffsx
a_y, b_y, c_y = coeffsy

a1 = (a_y+a_x) * pullyR / 4
a2 = (b_y+b_x) * pullyR / 4
a3 = (c_y+c_x) * pullyR / 4
b1 = (a_y-a_x) * pullyR / 4
b2 = (b_y-b_x) * pullyR / 4
b3 = (c_y-c_x) * pullyR / 4

print("Solved coefficients:")
print(pullyR)
print(f"a1 = {a1}")
print(f"a2 = {a2}")
print(f"a3 = {a3}")
print(f"b1 = {b1}")
print(f"b2 = {b2}")
print(f"b3 = {b3}")
# Calculate fitted Vx
Vx_fitted = a_x * x_dddot + b_x * x_ddot + c_x * x_dot
Vy_fitted = a_y * y_dddot + b_y * y_ddot + c_y * y_dot

# Calculate R-squared
ss_res_x = np.sum((Vx - Vx_fitted)**2)
ss_tot_x = np.sum((Vx - np.mean(Vx))**2)
r_squared = 1 - (ss_res_x / ss_tot_x)
print("X:")
print(f"\nR-squared: {r_squared:.6f}")
print(f"Root Mean Square Error: {np.sqrt(np.mean((Vx - Vx_fitted)**2)):.6f}")

ss_res_y = np.sum((Vy - Vy_fitted)**2)
ss_tot_y = np.sum((Vy - np.mean(Vy))**2)
r_squared = 1 - (ss_res_y / ss_tot_y)
print("Y:")
print(f"\nR-squared: {r_squared:.6f}")
print(f"Root Mean Square Error: {np.sqrt(np.mean((Vy - Vy_fitted)**2)):.6f}")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Vx actual vs fitted
axes[0, 0].plot(tx, Vx, 'b-', alpha=0.5, label='Actual Vx')
axes[0, 0].plot(tx, Vx_fitted, 'r-', linewidth=2, label='Fitted Vx')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Vx (PWM difference)')
axes[0, 0].set_title('Actual vs Fitted Vx')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Residuals
residuals = Vx - Vx_fitted
axes[0, 1].plot(tx, residuals, 'g-', alpha=0.7)
axes[0, 1].axhline(y=0, color='k', linestyle='--', linewidth=1)
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Residual')
axes[0, 1].set_title('Residuals (Actual - Fitted)')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Derivatives
#axes[1, 0].plot(t, x_dot, label="x'", alpha=0.7)
axes[1, 0].plot(tx, x_ddot, label="x''", alpha=0.7)
#axes[1, 0].plot(t, x_dddot, label="x'''", alpha=0.7)
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Derivative value')
axes[1, 0].set_title('Computed Derivatives of x')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Contribution of each term
contribution_a = a_x * x_dddot
contribution_b = b_x * x_ddot
contribution_c = c_x * x_dot
axes[1, 1].plot(tx, contribution_a, label=f"a·x''' ", alpha=0.7)
axes[1, 1].plot(tx, contribution_b, label=f"b·x''", alpha=0.7)
axes[1, 1].plot(tx, contribution_c, label=f"c·x'", alpha=0.7)
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Contribution to Vx')
axes[1, 1].set_title('Individual Term Contributions')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diffeq_fit_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nPlot saved as 'diffeq_fit_results.png'")