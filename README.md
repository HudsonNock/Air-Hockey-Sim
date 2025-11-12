**AI Air Hockey**

**Introduction**

Our goal is to deploy a neural network based agent to play air hockey against a human at a professional level. Although we do not have access to a professional air hockey player, the current version already outpreforms all humans that have played against it. To acheive this level of preformance, we designed and statistically modeled the real preformance of the table, including the vision accuracry, motor responses to voltage, timing throughout the firmware, and puck dynamics - while adjusting the electromechanical system so our models become more accurate. Using this model we wrote our own vectorized simulation which runs at 450x real time speed to train a reinforcement learning (RL) agent which was then deployed on the physcial air hockey table.

The first half of the project (8 month period) and its full in depth report is linked below. This covers the simulation design, vision calibrations, and reinforcement learning setup. However, note that much of the content in this final report has since been improved, including the reinforcement learning state space and reward function, as well as puck collision dynamics in the simulation.
[First Half Final Report](docs/2509 - AI Air Hockey - Final Report (PDF).pdf)

Video of RL agent playing against itself in the simulation:


Video of Human playing against the AI:


![air_hockey_cliponline-video-cutter com-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/45868e2b-58df-49db-9185-147e5af6fca6)
(low framerate due to GIF)

Below we give short outlines for each system, the technical details are outlined in the final report.

<details>
<summary>⚙️ Electro-Mechanical System</summary>

This project is part of a multiyear effort. Our team **inherited the electromechanical subsystem** from the previous group — this was the only major component we retained — though we made several adjustments and improvements.

The system consists of a **Core-XY gantry** mounted on a custom wooden **air hockey table** (approximately **1 m × 2 m**) that covers half the table’s surface.  
The gantry is driven by **two motors with timing belts**, each connected to **motor drivers** and controlled via an **STM32 “Blue Pill” microcontroller**. Both motors include **encoders** connected over **SPI** for feedback.

During aggressive motions, we observed **power-supply voltage sag**, leading to nonlinear behavior that made the system difficult to model accurately for simulation.  
To mitigate this, the previous team installed a **165 F supercapacitor** across the power rails to stabilize voltage under load.  
We later developed and documented a **safety procedure** for capacitor handling — [see here](link).

---

### ⚠️ Mechanical & Electrical Issues

- **Mallet carriage height variation** — The cables attached to the carriage run *above* the sliding beam, creating a bending moment.  
  This causes the mallet’s height to vary across the table. Since the mallet is positioned to avoid touching the surface to avoid friction, this bend in teh beam makes it possible for the **puck to be trapped under the malelt** when it is on the sides. Rapid belt tension changes also introduce **vertical vibration** and significant **audible noise** as the carriage impacts the table.  
  *(include image)*

- **Table is not rectangular** — The table is not a perfect rectangle, with the width changing by around 4 mm. This causes more variation in the puck dynamics as the simulation assumes it is a rectangle.

- **Table flatness** — The table surface itself is not perfectly flat, complicating **camera calibration** and essentuating the problem with the mallet carriage height variation.

- **Wooden walls** — The table’s walls are made of wood rather than plastic, producing **a high variance in measured collision dynamics** with location dependence. We noticed the wood also increases the chance of **puck ejection**.

- **Carriage looseness** — The original carriage was poorly constrained, causing **rattling**.  
  To resolve this, *Ian* redesigned the carriage to constrain all degrees of freedom and added slots for **shim adjustment**, enabling **≈50 µm precision** after 3D printing.  
  *(include images)*

- **Electrical interference** — The electrical system produced large amount of E&M noise, causing **serial communication and motor encoder noise**, resulting in incorrect bytes being transferred. We replaced these with **shielded cables**, eliminating the issue.

- **Supercapacitor charge circuit** — The original **pre-charge and discharge resistors** were oversized, resulting in **hour-long wait times**. We replaced them with **1 Ω power resistors**, reducing delays dramatically.

- **Roller geometry** — The rollers guiding the timing belts have a **bend radius that is too small** for the belt specification.  
  This causes premature belt wear, reduced tracking accuracy, and increased mechanical noise during motion.

- **Mallet Material** — The robot's mallet was originally printed out of PLA, however the impacts with the puck were too powerful and ended up breaking the mallet. We reprinted the mallet with PETG, high infill, and cuboid structure which has proven to be more impact resistant.

---

### 🚀 Future Steps

To improve system reliability, accuracy, and maintainability, several mechanical and structural upgrades are planned:

- **Upgrade to a professional-grade air hockey table**. Currently the table we have can't truley be considered air hockey as it doesn't meet regulation standards and has very uncertain dynamics. If we were to get a pro player to compete against our AI, we would want a level, uniform table, with plastic sides and consistent dynamics.

- **Redesign the gantry beam and cable routing** to prevent bending caused by the belt and cable geometry.  
  Possible solutions include repositioning the cable guides below the beam or reinforcing the beam with lightweight carbon-fiber or aluminum bracing.

- **Re-engineer the roller assemblies** with a **larger bend radius** or replace them with **idler pulleys** that match the belt’s minimum recommended curvature.  
  This will reduce mechanical stress, vibration, and belt fatigue.

- **Change mallet material**. Instead of 3D printing the mallet, we would like to attach a standard air hockey mallet to the mallet carriage, allowing for better collision dynamics.

These improvements would provide a more robust and consistent physical platform for reinforcement learning experiments, simulation validation, and long-term autonomous gameplay testing.

</details>


<details>
<summary>🎯 Computer Vision</summary>

Our computer vision system enables precise tracking of the puck and opponent mallet using a single camera placed on a tripod.

### Lighting and Puck Detection

Previous teams struggled to detect the puck without modifying it (e.g., adding LEDs).  
We solved this by applying **retroreflective material** to the puck and mounting a **high-intensity LED array** coaxially with the camera.  
This setup reflects light directly back to the lens, producing a **high-contrast puck image** even at 100 µs exposure, eliminating motion blur during high-speed play.

### Camera Calibration

Because the air hockey table was slightly **non-planar**, standard calibration techniques failed.  
We developed a **multi-view optimization procedure** that simultaneously solved for:
- The **3D positions of ArUco markers**, and  
- The **table surface height**, modeled as a second-order polynomial.  

Once the optimization proceedure was complete, we could then place the camera anywhere we desired and run a standard calibration technique to get the projection matrix. This approach enabled accurate mapping from image coordinates to real-world positions, achieving **~1 mm mean error** across the table. 

### Puck Tracking and Occlusion Handling

When the puck passes beneath the gantry, partial occlusion prevents simple centroid detection.  
To address this, we implemented a **contour-based tracking algorithm** that:
1. Projects visible puck contours to world space  
2. Generates multiple center candidates from the contour shape  
3. Selects the most consistent candidate based on geometric conformity  

This maintains reliable tracking even when a majority of the puck is hidden by the structure. There are instances where the puck is fully occluded, in these cases we set the puck position to the previous puck location instead of trying to estimate the position, this is implemented into the simulation allowing the RL agent to do the prediction.

### Opponent Mallet Tracking

We designed a lightweight **retroreflective mallet attachment** with a hollow center, producing a distinct contour from the puck.  
Two attachment variants support different playing styles (standard and low-profile).  
By detecting concentric contours, we can differentiate between puck and mallet and estimate both positions using the same calibrated camera system.

### 🏅 Key Achievements
- Achieved **robust, high-speed puck tracking** with **sub-millimeter precision**
- Developed a **calibration pipeline** resilient to table warping and uneven surfaces  
- Enabled **single-camera tracking** of both puck and mallet at 120 FPS  
- Fully passive optical system — **no modification to the puck or table electronics**

---

For a full description of the calibration math, optimization routine, and occlusion-robust tracking algorithm, see the [Final Report](link-to-report).

</details>

<details>
<summary>🧩 System ID</summary>

- <details>
  <summary>Mallet System ID</summary>

  ## Mallet System Identification

  To accurately simulate the environment, we needed to model the **mallet and puck dynamics**.  
  The mallet motion can be characterized as a **third-order transfer function** relating motor voltage to mallet position

   ```math
   \begin{bmatrix}
   V_1 \\
   V_2
   \end{bmatrix}
   =
   \begin{bmatrix}
   a_1 & a_2 & a_3 & b_1 & b_2 & b_3 \\
   b_1 & b_2 & b_3 & a_1 & a_2 & a_3
   \end{bmatrix}
   \begin{bmatrix}
   T_1^{***} \\
   T_1^{**} \\
   T_1^{*} \\
   T_2^{***} \\
   T_2^{**} \\
   T_2^{*}
   \end{bmatrix}
   ```

  We can then map this into two SISO systems, with the Cartesian control voltages as:

  $$
  V_y = -V_1 - V_2 = \frac{2}{R} \left[ (a_1 + b_1)\dddot{y} + (a_2 + b_2)\ddot{y} + (a_3 + b_3)\dot{y} \right]
  $$

  $$
  V_x = V_1 - V_2 = \frac{2}{R} \left[ (a_1 - b_1)\dddot{x} + (a_2 - b_2)\ddot{x} + (a_3 - b_3)\dot{x} \right]
  $$

  ### Parameter Identification

  Using encoder data for position and measured voltages \( V_x, V_y \), we performed parameter identification in MATLAB.

  The identification process involved:
  1. Splitting the trajectory data into **short path segments**.  
  2. For each segment, fitting a small polynomial to obtain the initial conditions.  
  3. Running an optimization over parameters \( a_1, ..., b_3 \) to minimize the mean squared error between simulated and measured motion.

  Only segments with a strong polynomial fit were used in the optimization.

  ---
  
  ### Feedforward and Feedback Control
  
  With the identified transfer function, we implemented **feedforward control** — generating voltage profiles that would ideally produce a desired mallet trajectory.
  
  However, due to nonlinearities (e.g., friction, backlash, and voltage saturation), the pure feedforward model was insufficient.  
  We therefore added **feedback control** with PID.
  
  To find the optimal feedback coefficients for PID in simulation:
  - We tuned the PID controller to follow x - x^ (the actual minus expected path) from data collected
  - Feedback voltages were then mapped to a change in position using the feedforward model previously identified
  - The loop modeled realistic factors such as voltage limits, delay between control updates, and the Blue Pill microcontroller’s control period.
  
  ---

  ### Results
  
  Combining feedforward and feedback control yielded **millimeter-level tracking accuracy** relative to the reference trajectory.
  
  This model forms the foundation of the simulated environment and ensures that reinforcement learning agents experience realistic, physics-based dynamics.
  
  </details>

- <details>
  <summary>Puck System ID</summary>
  
  ## Puck ODE

  To accurately simulate the air hockey environment, we needed to model the puck dynamics and collision behavior.  
  The puck motion can be described by a simple nonlinear ordinary differential equation:


  <p align="center">$m \ddot{x} = -f - B \dot{x}^2$</p>


  where  
  - \( m \) is the puck mass,  
  - \( f \) represents friction, and  
  - \( B \) is a drag coefficient term related to air resistance.
  
  We fit this model to the motion data obtained from tracking the puck. The parameters were estimated using nonlinear regression to minimize the mean squared error between the observed and predicted trajectories.

  ---
  
  ### Modeling Collisions
  
  Modeling the puck’s collisions with both the mallet and the walls proved more complex.  
  Initially, we modeled collisions using two separate restitution relationships:
  - Normal restitution as a function of incoming normal velocity, and  
  - Tangential restitution as a function of incoming tangential velocity.
  
  However, the real data exhibited significant variation, suggesting that the normal and tangential components were **not independent**.
  
  To better capture the behavior, we modeled the **output velocity and angle** as a function of the **input velocity and impact angle**.  
  Since no simple analytical function fit the data well, we instead trained a small neural network with **64 parameters** and **Softplus activations** to approximate the mapping.  
  
  The network also produced **heteroscedastic outputs**, meaning it learned to predict both the expected value and the uncertainty (standard deviation) for each output dimension.
  
  ---
  
  ### Data Processing
  
  The collision dataset was built from filtered puck trajectories:
  1. We first identified segments where the puck trajectory could be fit accurately by a linear model over time.  
  2. Adjacent linear segments were extrapolated to detect potential intersections — either with a wall or a mallet.
  3. If the intersection occurred near a wall, it was labeled as a **wall collision**; otherwise, if it occurred near the mallet position, it was labeled as a **puck–mallet collision**.
  
  For each mallet collision:
  - The mallet trajectory was fitted using a polynomial within a 30 ms window.  
  - The exact contact point was found by minimizing  
    $$
    \| P_\text{puck} - P_\text{mallet} \| - (r_\text{puck} + r_\text{mallet})
    $$
    where \( P_\text{puck} \) and \( P_\text{mallet} \) are positions, and \( r_\text{puck} \), \( r_\text{mallet} \) are radii.
  
  We then transformed all collision data into the **mallet frame of reference** and computed:
  - Incoming velocity and angle  
  - Outgoing velocity and angle
  
  This produced a large dataset used to train the neural network model.
  
  ---
  
  ### Extrapolation for Out-of-Distribution Data
  
  Since we could not experimentally capture high-velocity collisions, we augmented the dataset with **synthetic extrapolated samples** at higher speeds.  
  This ensured that the simulation remained stable and physically reasonable even in scenarios that extended beyond the training distribution.
  
  ---
  
  ### Summary
  
  The resulting model captures both deterministic and stochastic aspects of puck dynamics:
  - The ODE models continuous motion under drag and friction.  
  - The neural network models nonlinear, uncertain collision responses.  
  
  Together, these models provide a realistic simulation of puck behavior suitable for reinforcement learning and physics-based gameplay.
  </details>

</details>

<details>
<summary>Firmware and Timings</summary>



</details>

<details>
<summary>Simulation</summary>

</details>

<details>
<summary>Reinforcment Learning</summary>

</details>



**Completed:**

**Simulation**: Coded and vectorized an air hockey simulation which runs based on a transfer function of the real world table dynamics (relating motor voltages to position) generated though collecting real world data. The simulation works by using numerical methods to solve equations of intersection between the puck and the mallet, as well as the puck and the wall.

**Action Space**: The robot outputs final position and curve hyperparameters, we then generate a path using the hyperparameters that converges to the final position and has a corresponding voltage max less than our supply voltage. This action space was chosen since it gives a lot of freedom to the agent, yet it is impossible for the agent to perform a sequence of actions resulting in a collision with the table walls. However, there is some degeneracy in the action space when the final position is sufficiently far away from its initial position, so this action space may still be subject to change.

**State Space**: Each agent is given its, the opponents, and the puck's position and velocity, as well as the time that has passed since the puck entered its side of the table. We also give the agent the expected position and velocity of the puck in the next time step assuming we ignore mallets (I suspect this is hard for a small NN to calculate since multiple wall hits can occur in one time step resulting in long variable length sequential calculations).

**Reward**: Modified the reward function so the agent gets better at scoring even after defense was perfected. This is done by determining the trajectory of the puck in the next second after it crosses the centerline, and if its trajectory enters the opponents goal it receives a reward. Additionally, to speed up learning, a termination condition is met if the puck stays on the agent's side for more than 5 seconds.

**Training**: Ran several multi agent environments in parallel and trained using PPO implemented by torchRL.


**To Add:**
* Noise and domain randomization (already added in the simulation, but currently training without it for a baseline result)
* Tune the reward structure and hyperparameters until the agent acts perfectly (potentially increase the model size). Modifications to the reward structure could also incentivise the agent to keep the puck at a slow speed when the puck is on the agent's side, which would help perfect a goal scoring trajectory.
* Add a delay to the agents actions to better simulate the real world dynamics. Additionally, to minimize delay on the physical application, another action could be added that preserves the previous action and steps forward some small dt < Ts (sampling time of the agent). This would reduce the total reaction time by at most Ts, however, this setup would also take longer to train due to how the simulation was implemented.
* Run the agent on the physical table
