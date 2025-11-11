import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
table_width = 1.9885       # meters
table_height = 0.9905      # meters
puck_radius = 0.035306     # meters
window = 4                 # frames to average for restitution
m = 5                      # frame lag for dv threshold check
max_dv = 0.2               # m/s change over m frames triggers new interval
min_interval_length = 10  # minimum frames per interval to keep
tol = 0.1                 # proximity tolerance for wall detection

# Load data
vn_coeffs, vt_coeffs = [], []
vn_init, vt_init, v_total_init = [], [], []
for file_i in range(2):
    df = pd.read_csv(f'position-mar-14-{file_i + 1}.csv')
    df['dt'] = df['dt'].fillna(df['dt'].mean())
    df['t'] = df['dt'].cumsum()

    # Compute velocities via central difference
    def compute_velocity(arr, t, w):
        v = np.full_like(arr, np.nan, dtype=float)
        for i in range(w, len(arr)):
            dt_total = t[i] - t[i - w]
            v[i] = (arr[i] - arr[i - w]) / dt_total
        return v

    df['vx'] = compute_velocity(df['x'].values, df['t'].values, window)
    df['vy'] = compute_velocity(df['y'].values, df['t'].values, window)

    # Build intervals separated by large velocity jumps
    dv = np.hypot(df['vx'] - df['vx'].shift(m), df['vy'] - df['vy'].shift(m))
    intervals = []
    current = []
    for idx in df.index:
        if idx >= m and dv.loc[idx] > max_dv:
            # end current interval
            if current:
                intervals.append(current)
            current = [idx]
        else:
            current.append(idx)
    # append the last
    if current:
        intervals.append(current)

    # Filter out short intervals
    intervals = [iv for iv in intervals if len(iv) >= min_interval_length]

    # Helper to detect wall type at a given index
    def wall_type_at(idx):
        x, y = df.loc[idx, ['x', 'y']]
        near_left = x <= (puck_radius + tol)
        near_right = False #x >= (table_width - puck_radius - tol)
        near_bottom = y <= (puck_radius + tol)
        near_top = y >= (table_height - puck_radius - tol)
        if near_left or near_right:
            return 'vertical_wall'
        if near_top or near_bottom:
            return 'horizontal_wall'
        return None

    # Decompose velocity vector
    def decompose(vx, vy, wt):
        if wt == 'vertical_wall':
            return vx, vy
        else:
            return vy, vx

    # Compute restitution across interval boundaries

    for k in range(len(intervals) - 1):
        iv0, iv1 = intervals[k], intervals[k + 1]
        # midpoint index for wall detection
        mid_idx = iv1[0]
        wt = wall_type_at(mid_idx)
        if wt is None:
            continue
        # average last and first window velocities
        v0 = df.loc[iv0[-window:], ['vx', 'vy']].mean().values
        v1 = df.loc[iv1[:window], ['vx', 'vy']].mean().values
        if np.linalg.norm(v0) < 0.2 or np.linalg.norm(v1) < 0.2:
            continue
        vn0, vt0 = decompose(v0[0], v0[1], wt)
        vn1, vt1 = decompose(v1[0], v1[1], wt)
        if vn0 == 0:
            continue
        e_n = -vn1 / vn0
        e_t = vt1 / vt0 if vt0 != 0 else np.nan
        if e_n > 0 and e_n < 1 and e_t > 0 and e_t < 1:
            vn_coeffs.append(e_n)
            vt_coeffs.append(e_t)
            vn_init.append(abs(vn0))
            vt_init.append(abs(vt0))
            v_total_init.append(np.hypot(v0[0], v0[1]))
        else:
            continue

results = pd.DataFrame({
    'e_normal': vn_coeffs,
    'e_tangent': vt_coeffs,
    'v_normal': vn_init,
    'v_tangent': vt_init,
    'v_total': v_total_init
})

# Plotting
# 1. vs total speed
plt.figure()
plt.scatter(results['v_total'], results['e_normal'], alpha=0.7)
plt.xlabel('Initial Total Speed (m/s)')
plt.ylabel('Normal Restitution Coefficient')
plt.title('e_normal vs Total Speed')
plt.grid(True)
plt.show()

plt.figure()
plt.scatter(results['v_total'], results['e_tangent'], alpha=0.7)
plt.xlabel('Initial Total Speed (m/s)')
plt.ylabel('Tangential Restitution Coefficient')
plt.title('e_tangent vs Total Speed')
plt.grid(True)
plt.show()

# 2. vs normal component
plt.figure()
plt.scatter(results['v_normal'], results['e_normal'], alpha=0.7)
plt.xlabel('Initial Normal Speed (m/s)')
plt.ylabel('Normal Restitution Coefficient')
plt.title('e_normal vs Normal Speed')
plt.grid(True)
plt.show()

# 3. vs tangential component
plt.figure()
plt.scatter(results['v_tangent'], results['e_tangent'], alpha=0.7)
plt.xlabel('Initial Tangential Speed (m/s)')
plt.ylabel('Tangential Restitution Coefficient')
plt.title('e_tangent vs Tangential Speed')
plt.grid(True)
plt.show()