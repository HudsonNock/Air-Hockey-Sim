import numpy as np
import matplotlib.pyplot as plt
import control as ctl

pullyR = 0.035755
a1 = 2.725e-05 #7.474*10**(-6) 
a2 = 7.575e-03 #6.721*10**(-3) 
a3 = 6.969e-02 #6.658*10**(-2)
b1 = -1.996e-05 #-1.607*10**(-6)
b2 = -2.838e-03 #-2.731*10**(-3)
b3 = 3.688e-03 #3.610*10**(-3)

a = 2*(2.725e-05 -1.996e-05) / pullyR
b = 2*(7.575e-03 -2.838e-03)/pullyR
c = 2*(6.969e-02 + 3.688e-03)/pullyR

# PID controller parameters (tune these)
Kp = 100.0 #25.0
Ki = 3.4 #10.0
Kd = 8 #3.0

# Saturation limits
u_min, u_max = -12.0, 12.0

# Define continuous-time plant
G = ctl.TransferFunction([1.0], [a, b, c, 0])

# Discretize the plant for numerical simulation
dt = 0.001
Gd = ctl.sample_system(G, dt)

# Extract discrete system matrices (state-space form)
sysd = ctl.tf2ss(Gd)
A, B, C, D = sysd.A, sysd.B, sysd.C, sysd.D

# Simulation setup
t_final = 10
t = np.arange(0, t_final, dt)
r = np.ones_like(t)  # unit step reference
x = np.zeros((A.shape[0],))  # system state
y = 0.0
u = 0.0

# PID states
integral = 0.0
prev_error = 0.0

# Logs
y_log = []
u_log = []

for i in range(len(t)):
    # Error
    e = r[i] - y

    # PID controller
    integral += e * dt
    derivative = (e - prev_error) / dt
    u = Kp * e + Ki * integral + Kd * derivative

    # Saturation
    u = np.clip(u, u_min, u_max)

    # Update system state (discrete-time simulation)
    x = A @ x + B.flatten() * u
    y = C @ x + D.flatten() * u
    y = float(y)

    # Save values
    y_log.append(y)
    u_log.append(u)
    prev_error = e

# Plot results
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, y_log, label="System Output y(t)")
plt.plot(t, r, 'k--', label="Reference r(t)")
plt.title("Closed-loop Step Response with PID Saturation")
plt.xlabel("Time [s]")
plt.ylabel("Output")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, u_log, 'r', label="Controller Output u(t)")
plt.title("PID Controller Output (Capped at ±12 V)")
plt.xlabel("Time [s]")
plt.ylabel("Control Effort [V]")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
