import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Load CSV ===
df = pd.read_csv(
    "mallet_data_NN_paths2.csv"
)[:32400]

print(df["dt"][11422:11428])

pullyR = 0.035755
print(pullyR * np.pi)
print(0.7706 - 0.7065)


df["Vx"] = (df["pwm_x"] - df["pwm_y"]) * 24
df["Vy"] = (-df["pwm_x"] - df["pwm_y"]) * 24

# If 'dt' is a timestep, create cumulative time
time = df["dt"].cumsum()

# === X plot ===
fig, ax1 = plt.subplots(figsize=(10, 5))

# Primary y-axis for position
ax1.plot(time, df["x"], label="x", color="tab:blue")
ax1.plot(time, df["Expected_x"], label="Expected_x", linestyle="--", color="tab:orange")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Position (X)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.grid(True)

# Secondary y-axis for PWM
ax2 = ax1.twinx()
ax2.plot(time, df["Vx"], label="Vx", linestyle=":", color="tab:green")
ax2.set_ylabel("PWM (X)", color="tab:green")
ax2.tick_params(axis="y", labelcolor="tab:green")

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.title("X and PWM X vs Time")
plt.tight_layout()

# === Y plot ===
fig, ay1 = plt.subplots(figsize=(10, 5))

# Primary y-axis for position
ay1.plot(time, df["y"], label="y", color="tab:blue")
ay1.plot(time, df["Expected_y"], label="Expected_y", linestyle="--", color="tab:orange")
ay1.set_xlabel("Time (s)")
ay1.set_ylabel("Position (Y)", color="tab:blue")
ay1.tick_params(axis="y", labelcolor="tab:blue")
ay1.grid(True)

# Secondary y-axis for PWM
ay2 = ay1.twinx()
ay2.plot(time, df["Vy"], label="Vy", linestyle=":", color="tab:green")
ay2.set_ylabel("PWM (Y)", color="tab:green")
ay2.tick_params(axis="y", labelcolor="tab:green")

# Combine legends
lines1, labels1 = ay1.get_legend_handles_labels()
lines2, labels2 = ay2.get_legend_handles_labels()
ay1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.title("Y and PWM Y vs Time")
plt.tight_layout()
plt.show()

"""
# If 'dt' is a timestep, create cumulative time
time = df["dt"].cumsum()

df["t1"] = (df["x"] - df["y"]) * 16384 / (2*np.pi*pullyR)
df["t2"] = (-df["x"] - df["y"]) * 16384 / (2*np.pi*pullyR)
# === X plot ===
fig, ax1 = plt.subplots(figsize=(10, 5))

# Primary y-axis for position
ax1.plot(time, df["t1"], label="t1", color="tab:blue")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("theta 1", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.grid(True)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(lines1, labels1, loc="best")

plt.title("X and PWM X vs Time")
plt.tight_layout()

# === Y plot ===

fig, ay1 = plt.subplots(figsize=(10, 5))

# Primary y-axis for position
ay1.plot(time, df["t2"], label="t2", color="tab:blue")
ay1.set_xlabel("Time (s)")
ay1.set_ylabel("theta 2", color="tab:blue")
ay1.tick_params(axis="y", labelcolor="tab:blue")
ay1.grid(True)


# Combine legends
lines1, labels1 = ay1.get_legend_handles_labels()
ay1.legend(lines1, labels1, loc="best")

plt.title("Y and PWM Y vs Time")
plt.tight_layout()
plt.show()
"""