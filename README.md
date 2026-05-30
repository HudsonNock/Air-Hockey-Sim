# AI Air Hockey

**Project Goal**

Our goal is to close the sim to real gap and deploy an AI model to successfully play air hockey against a human. Below is a video of our end results with a quick overview (click image to play video):

[![Watch the video](docs/gameplay_video_screenshot.png)](https://www.youtube.com/watch?v=75mm_B950PQ)


System Overview:

![System Diagram:](docs/air_hockey_flowchart.png)

**Challenges**

To achieve this level of performance, we designed and statistically modeled the real performance of the table, including the vision accuracy, motor responses to voltage, timing throughout the firmware, and puck dynamics - while adjusting the electromechanical system so our models become more accurate. Using this model we wrote our own vectorized simulation to train a reinforcement learning (RL) agent which was then deployed on the physical air hockey table.

Below is a video of the agent playing against itself in simulation.

![](docs/195_vid.gif)

**Report and Details**

The first half of the project (the first 8 months) and its full in depth report is linked below. This covers the simulation design, vision calibrations, and reinforcement learning setup. The second half of the project (last 8 months) is also included, covering the firmware, system dynamics, puck collision dynamics, reinforcement learning updates, and mechanical improvements. The project was spaced with a four month break between the two halves.

First Half of Project: [Final Report](docs/2509-AIAirHockey-FinalReport.pdf) <br>
Second Half of the Project: [Final Report](docs/2550__AI_Air_Hockey_Final_Report.pdf)

Below we give short outlines for each system, many of the technical details are outlined in the final report. Note that this repo contains all of my personal code across the 4 branches, see https://github.com/AIAirHockey for mallet system ID code and code for puck motion differential equation fitting.

## Table of Contents

* [⚙️ Electro-Mechanical System](#%EF%B8%8F-electro-mechanical-system)
* [🎯 Computer Vision](#-computer-vision)
* [🧩 Mallet System ID](#-mallet-system-id)
* [🔘 Puck System ID](#-puck-system-id)
* [💻 Firmware and System Timings](#-firmware-and-system-timings)
* [🎮 Simulation](#-simulation)
* [🧠 Reinforcement Learning](#-reinforcement-learning)

# ⚙️ Electro-Mechanical System
[table of contents](#table-of-contents)

This project is part of a multiyear effort. Our team **inherited the electromechanical subsystem** from the previous group, though we ended up redesiging most of the mechanical system.

The original system consists of a **Core-XY gantry** mounted on a custom wooden air hockey table (approximately **1 m × 2 m**) that covers half the table’s surface.  
The gantry is driven by **two motors with timing belts**, each connected to **motor drivers** and controlled via an **STM32 “Blue Pill” microcontroller**. Both motors include **encoders** connected over **SPI** for feedback.

![](docs/IMG_20251113_114419409.jpg)

During aggressive motions, we observed **power-supply voltage sag**, leading to nonlinear behavior that made the system difficult to model and control.  
To mitigate this, the previous team installed a **165 F supercapacitor** across the power rails to stabilize voltage under load.  
We later developed and documented a **safety procedure** for capacitor handling — [see here](https://docs.google.com/presentation/d/1C2lxZXDaFv2uMI581Z3ULeYxlSjX5ojVwfU028PKIhc/edit?usp=sharing).

---

### ⚠️ Mechanical & Electrical Issues

- **Mallet carriage height variation** — The cables attached to the carriage run *above* the sliding beam, creating a bending moment.

  ![](docs/IMG_20251113_114357875~2.jpg)
  
  This causes the mallet’s height to vary across the table. Since the mallet is positioned to avoid touching the surface to avoid friction, this bend in the beam makes it possible for the **puck to be trapped under the mallet**. Rapid belt tension changes also introduce **vertical vibration** and significant **audible noise** as the carriage impacts the table.

- **Table is not rectangular** — The table is not a perfect rectangle, with the width changing by around 4 mm. This causes more variation in the puck dynamics as the simulation assumes it is a rectangle.

- **Table flatness** — The table surface itself is not perfectly flat, complicating **camera calibration** and accentuating the problem with the mallet carriage height variation.

- **Wooden walls** — The table’s walls are made of wood rather than plastic, producing **high variance** in measured collision dynamics with location dependence. We noticed the wood also increases the chance of **puck ejection**.

- **Carriage looseness** — The original carriage was poorly constrained, causing **rattling**.  
  To resolve this, Ian redesigned the carriage to constrain all degrees of freedom and added slots for **shim adjustment**, enabling **≈50 µm precision** after 3D printing.  

  ![](docs/new_mallet_carriage.png)

- **Electrical interference** — The electrical system produced large amount of E&M noise, causing **serial communication and motor encoder noise**, resulting in incorrect bytes being transferred. We replaced these with **shielded cables**, eliminating the issue.

- **Supercapacitor charge circuit** — The original **pre-charge and discharge resistors** were oversized, resulting in **hour-long wait times**. We replaced them with **1 Ω power resistors**, reducing delays dramatically.

- **Roller geometry** — The rollers guiding the timing belts have a **bend radius that is too small** for the belt specification.

- **Mallet Material** — The robot's mallet was originally printed out of PLA, however the impacts with the puck were too powerful and ended up breaking the mallet. We reprinted the mallet with PETG, high infill, and cuboid structure which has proven to be more impact resistant.

- **Motor encoder mounts** — After hours of play, the motors would heat up enough that the 3D printed encoder mounts would warp resulting in inaccurate readings.

---

### 🚀 Upgrades

After integrating everything on the wooden table with a successful zero shot sim to real transfer, our next goal was to get a pro player to compete against it. However, the wooden table doesn't meet regulation standards and has very uncertain dynamics, leading us to focus on **upgrading to a professional-grade air hockey table**, this involved:

- **Designing a frame**. We went with a 80/20 frame design that allowed easy mounting of components as well as adjustable air hockey table dimensions (either 7 ft or 8 ft table surface). We calculated the maximum theoretical deformation under the applied static load and designed it to be within 0.2 mm ensuring a flat playing surface.

![](docs/IMG_20260214_193535812.jpg)

![](docs/IMG_20260521_164919802.jpg)

- **Redesign the gantry beam and cable routing** to prevent bending of the crossbeam. To achieve this, we placing the belt along the central axis of the beam and increased the moment of inertia of the beam. Ian chose the new crossbeam and designed a new mallet carriage appropriately. Additionally, I redesign other pulley blocks to fit in the new layout.

![](docs/IMG_20260521_165110532.jpg)

- **Re-engineer the roller assemblies** with a **larger bend radius** or replace them with **idler pulleys** that match the belt’s minimum recommended curvature.

- **Change mallet material**. Instead of 3D printing the mallet, Ian attached a standard air hockey mallet to the mallet carriage, allowing for better collision dynamics.

- **Change motor encoder mounts**. These were redesigned by Mauro to include thermally insulating standoffs allowing for no warping after play.

### Future Challanges

One main concern of the current mechanical design is the dynamic tension that is felt periodically when moving the gantry around indicating that the system is not LTI. We suspect this is due to the trapazoidal belt profile but have not had time to resolve the issue.

# 🎯 Computer Vision
[table of contents](#table-of-contents)

Our computer vision system enables precise tracking of the puck and opponent mallet using a single camera placed on a tripod.

### Lighting and Puck Detection

Previous teams struggled to detect the puck without modifying it (e.g., adding LEDs).  
We solved this by applying **retroreflective material** to the puck and mounting a **high-intensity LED array** coaxially with the camera.  
This setup reflects light directly back to the lens, producing a **high-contrast puck image** even at 100 µs exposure, eliminating motion blur during high-speed play.

Camera View:

![](docs/camera_view.png)

Retroreflective Tape Diagram, visualizing how it reflects light directly back to the source:

![](docs/retroreflective_tape_diagram.png)

### Camera Calibration

Standard calibration techniques are to determine the camera's intrinsic matrix and use the measured location of ArUco marker position to determine the camera's location and orientation in space. Below is the camera's view where the 6 markers around the edge of the table being the ArUco markers:

![](docs/IMG_20251113_151841704.jpg)

However, as the table is non-planar and large, we can not precisely measure the exact position of the ArUco markers and so standard calibration techniques failed.  
Instead we developed a **multi-view optimization procedure** that simultaneously solved for:
- The **3D positions of ArUco markers**, and  
- The **table surface height**, modeled as a second-order polynomial.  

Once the optimization procedure was complete, we could then place the camera anywhere we desired and run a standard calibration technique to get the projection matrix. This approach enabled accurate mapping from image coordinates to real-world positions, achieving **~1 mm mean error** across the table. Calibration details are provided in the 2509 final report with updates in 2550 final report.

### Puck Tracking and Occlusion Handling

When the puck passes beneath the gantry, partial occlusion prevents simple centroid detection.  
To address this, we implemented a **contour-based tracking algorithm** that:
1. Projects visible puck contours to world space  
2. Generates multiple center candidates from the contour shape  
3. Selects the most consistent candidate based on geometric conformity  

This maintains reliable tracking even when a majority of the puck is hidden by the structure. There are instances where the puck is fully occluded, in these cases we set the puck position to the previous puck location instead of trying to estimate the position, this is also implemented into the simulation allowing the RL agent to do the prediction.

### Opponent Mallet Tracking

We designed a lightweight **retroreflective mallet attachment** with a hollow center, producing a distinct contour from the puck.  
By detecting concentric contours, we can differentiate between puck and mallet and estimate both positions using the same calibrated camera system.

### 🏅 Key Achievements
- Achieved **robust, high-speed puck tracking** with **sub-millimeter precision**
- Developed a **calibration pipeline** resilient to table warping and uneven surfaces  
- Enabled **single-camera tracking** of both puck and mallet at 120 FPS  
- Fully passive optical system — **no modification to the puck or table electronics**

---

For a full description of the calibration math, optimization routine, and occlusion-robust tracking algorithm, see the [Final Report](link-to-report).

# 🧩 Mallet System ID
[table of contents](#table-of-contents)

To accurately simulate the environment, we needed to model the **mallet and puck dynamics**.  
The mallet motion can be characterized as a **third-order transfer function** relating motor voltage to mallet position

![](docs/feedforward_eq.png)

We can then map this into two SISO systems, with the Cartesian control voltages as:

<p align="center">$V_y = -V_1 - V_2 = \frac{2}{R} \left[ (a_1 + b_1)\dddot{y} + (a_2 + b_2)\ddot{y} + (a_3 + b_3)\dot{y} \right]$</p>

<p align="center">$V_x = V_1 - V_2 = \frac{2}{R} \left[ (a_1 - b_1)\dddot{x} + (a_2 - b_2)\ddot{x} + (a_3 - b_3)\dot{x} \right]$</p>

### Parameter Identification

Using encoder data for position and measured voltages \( V_x, V_y \), we performed parameter identification in MATLAB. The code for this section was written by Mauro.

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

# 🔘 Puck System ID
[table of contents](#table-of-contents)

## Puck ODE

To accurately simulate the air hockey environment, we needed to model the puck dynamics and collision behavior.  
The puck motion can be described by a simple nonlinear ordinary differential equation:

<p align="center">$m \ddot{x} = -f - B \dot{x}^2$</p>

where  
- \( m \) is the puck mass,  
- \( f \) represents friction, and  
- \( B \) is a drag coefficient term related to air resistance.

We fit this model to the motion data obtained from tracking the puck. The parameters were estimated using nonlinear regression to minimize the mean squared error between the observed and predicted trajectories. The optimization code here was written by Ian.

---

### Modeling Collisions

Modeling the puck’s collisions with both the mallet and the walls proved more complex.  
Initially, we modeled collisions using two separate restitution relationships:
- Normal restitution as a function of incoming normal velocity, and  
- Tangential restitution as a function of incoming tangential velocity.

However, the real data exhibited significant variation, suggesting that the normal and tangential components were **not independent**.

To better capture the behavior, we modeled the **output velocity and angle** as a function of the **input velocity and impact angle**.  
Since no simple analytical function fit the data well, we instead trained a small neural network with **112 parameters** and **Softplus activations** to approximate the mapping.

![](docs/collision_NN_vel.png)
![](docs/collision_NN_angle.png)

The network also produced **heteroscedastic outputs**, meaning it learned to predict both the expected value and the uncertainty (standard deviation) for each output dimension.

![](docs/collision_NN_sigma.png)

---

### Data Processing

The collision dataset was built from filtered puck trajectories:
1. We first identified segments where the puck trajectory could be fit accurately by a linear model over time.  
2. Adjacent linear segments were extrapolated to detect potential intersections — either with a wall or a mallet.
3. If the intersection occurred near a wall, it was labeled as a **wall collision**; otherwise, if it occurred near the mallet position, it was labeled as a **puck–mallet collision**.

For each mallet collision:
- The mallet trajectory was fitted using a polynomial within a 30 ms window.  
- The exact contact point was found by minimizing
  <p align="center">$| \| P_\text{puck} - P_\text{mallet} \| - (r_\text{puck} + r_\text{mallet}) |$</p>

We then transformed all collision data into the **mallet frame of reference** and computed:
- Incoming velocity and angle  
- Outgoing velocity and angle

This produced a large dataset used to train the neural network model.

---

### Extrapolation for Out-of-Distribution Data

Since we could not experimentally capture high-velocity collisions, we augmented the dataset with **synthetic extrapolated samples** at higher speeds.  
This ensured that the simulation remained stable and physically reasonable even in scenarios that extended beyond the training distribution.

![](docs/collision_NN_out_of_dist.png)

---

### Summary

The resulting model captures both deterministic and stochastic aspects of puck dynamics:
- The ODE models continuous motion under drag and friction.  
- The neural network models nonlinear, uncertain collision responses.  

Together, these models provide a realistic simulation of puck behavior suitable for reinforcement learning and physics-based gameplay.

# 💻 Firmware and System Timings
[table of contents](#table-of-contents)

## System Architecture

The system runs on Ubuntu 22.04 with real-time kernel flags enabled, coordinating three main components:
- **Laptop**: Processes camera input and Blue Pill data, runs neural network
- **Blue Pill**: Reads mallet position and communicates with motor encoders via SPI
- **Camera**: Captures table state at 120 fps

## Serial Communication Protocol

### Blue Pill → Laptop (Mallet Position)

To minimize serial bandwidth while maintaining reliability, we implemented a custom encoding scheme:

- **Data Encoding**: Float values are scaled to `int16` range before transmission using calculated bounds for each parameter.
- **Packet Structure**: 
  - Start flag: `0xFF 0xFF 0xFF`
  - Data payload (scaled `int16` values)
  - End flag: `0x55`
  - Checksum: NAND of all data bytes

**Noise Resilience**: If data contains `0xFF`, it could be mistaken for part of the start flag. To prevent this, any least significant byte equal to `0xFF` is reduced to `0xFE`. This sacrifices at most 1 LSB (~50 microns), which is negligible compared to the reliability gain.

### Laptop → Blue Pill (Control Parameters)

Voltage curve parameters are sent via UART with `\n` as start/end delimiters, also including a NAND checksum. The Blue Pill buffers incoming serial data before processing.

## Camera Configuration

The camera streams BayerRG8 format with a reduced ROI to achieve 120 fps without exceeding USB 3.0 bandwidth limits. Grayscale was tested but requires on-camera RGB processing, significantly reducing achievable frame rates.

## Timing Measurements

Accurate delay measurement is critical for the simulation. Rather than measuring individual one-way delays (which requires clock synchronization), we use **round-trip timing**. The central idea here is not to measure the information delay from the event to when the neural network receives it, but instead the delay from the event to the neural networks action being preformed. In simulation, we can then treat the neural networks action as having no delay and set the information delay as the round trip delay.

### Mallet Delay
1. Blue Pill sends current timestamp (instead of dt)
2. Laptop processes through NN and path planning
3. Laptop echoes timestamp back in path variables
4. Blue Pill calculates: `current_time - received_timestamp = total_delay`

### Camera Delay
1. Blue Pill turns on LED with known timestamp
2. Laptop detects LED brightness in frame
3. Laptop processes preloaded table frame through entire pipeline
4. Blue Pill receives command and calculates delay
5. Result: Uniform distribution over (delay+[0, 1/120s]), where the lower bound is the actual camera information delay

All timing measurements include mean, standard deviation, min, and max values.

## Real-Time Performance

With code optimizations and the RT kernel, the main loop processes in **6-7 ms** (typical), though occasionally exceeds 8.5 ms. To prevent timing drift, the system runs at **60 Hz** instead of 120 Hz.

### Main Loop Sequence
```
1. Wait for camera data
2. Read mallet position from serial
3. Process NN action
4. Calculate Initial Condition based on previous command (accounting for command delay)
5. Calculate feedforward voltage profile, using initial condition and new command
6. Send instructions to Blue Pill
7. Capture next camera frame to buffer
8. Repeat at 60 Hz
```

### CPU Optimization
- Two CPU cores isolated for the main process
- CPU frequency locked to maximum
- Process pinned to isolated cores for consistent performance

# 🎮 Simulation
[table of contents](#table-of-contents)

## Overview

The simulation is implemented in Python with full vectorization support, enabling parallel execution of multiple environments. Unlike traditional physics simulations that use fixed timesteps, this implementation leverages analytical solutions for significantly improved performance.

## Motion Model

### Mallet Control
Mallets move along paths defined by initial conditions, desired final position, and system parameters `M_Vx` and `M_Vy`. Each path corresponds to a voltage profile (see System Identification section):

<p align="center">$V_x(t) = M_{V_x} u(t) - 2M_{V_x} u(t-t_1) + M_{V_x} u(t-t_2)$</p>

Given the final position and system parameters, we solve for `t₁` and `t₂` to minimize overshoot and converge to the final position, xf. This formulation guarantees:
- All generated paths are physically achievable within system constraints
- The mallet never collides with walls
- The ODE has an analytic solution which is preprogramed and so no compute is used solving it (this is one strong reason to code our own simulation compared to using a pre-existing one)

### Puck Dynamics
As derived in the System Identification section, puck motion satisfies a differential equation with a closed-form solution. Both mallet and puck positions are expressed as **explicit functions of time**—no iterative solvers required.

## Collision Detection

Collisions (puck-wall and puck-mallet) are detected by solving for when positions are the correct distance apart. While this lacks a closed-form solution, we:

1. Derive a **lower bound** on time-to-collision based on current actions and initial conditions (detailed in final report)
2. Step forward by this lower bound iteratively
3. Terminate when collision occurs or lower bound exceeds remaining simulation time

This approach guarantees no collisions are missed while maintaining large timesteps.

Collision responses are modeled using the heteroscedastic neural network described in the System Identification section.

## Performance Advantages

Traditional simulations use fixed `dt` timesteps, creating a trade-off between precision and speed. This implementation uses **dynamic timesteps** with several key optimizations:

1. **Analytical Solutions**: Both mallet and puck motion have closed-form solutions—no numerical integration needed
2. **Adaptive Timesteps**: High precision between collisions, large steps when positions change slowly
3. **Precomputed Paths**: Voltage profile restriction allows ODE solutions to be calculated once and reused
4. **Vectorized Execution**: Multiple environments run in parallel using NumPy operations

**Result**: Able to run the simulation at **450× real-time speedup** on an Intel i5 processor.

*Additional implementation details and derivations are provided in the final report.*

# 🧠 Reinforcement Learning
[table of contents](#table-of-contents)
  
# Markov Decision Process

## State Space

The agent receives a comprehensive state representation designed to handle real-world timing delays and sensor limitations:

### Puck History Buffer
A buffer of 5 historical puck positions at indices **0, 1, 2, 5, 11**, corresponding to approximate delays of:
- 8 ms
- 17 ms  
- 25 ms
- 50 ms
- 100 ms

**Rationale**: Direct velocity calculation is unreliable in certain scenarios (e.g., puck bouncing off corners). Providing raw positional history allows the network to implicitly infer velocity and acceleration.

### Opponent Information
Historical opponent mallet positions at the same buffer indices (0, 1, 2, 5, 11).

### Agent State
- **Mallet position**: Previous position (delayed—see Firmware and Timing section)
- **Mallet velocity**: Calculated from delayed position data
- **Previous action**: Ensures the MDP is fully defined, as `previous_action + delayed_position` determines current position

### Domain Parameters
Current system identification coefficients `(a₁, a₂, a₃, b₁, b₂, b₃)` being used (see system id section). These are included because they vary across training domains (see Domain Randomization section).

## Action Space

The agent outputs three continuous values:

- **`xf`**: Desired final mallet position
- **`M_{V_x}`**: Voltage parameter for x-axis motion
- **`M_{V_y}`**: Voltage parameter for y-axis motion

These define a trajectory as described in the Simulation Implementation section, guaranteeing physically feasible paths within system constraints.

## Reward Function

### Initial Approach (Sparse Rewards)
Initially, rewards were:
- **+1** if puck enters opponent's goal
- **-1** if agent gets scored on

**Problem**: Defense is easier than offense, making rewards extremely sparse and hindering learning.

### Improved Approach

When the puck crosses the halfway line toward the opponent:

1. **Rollout Simulation**: Simulate the puck trajectory, assuming the opponent remains stationary
2. **Reward Calculation**:
   Give an initial award of
   ```
   reward = A * puck_velocity
   ```
   Where A = 11.5 if the puck trajectory enters the opponents goal avoiding the opponent, 2.5 if it enters the opponents goal when ignoring the opponent, and 0 if it doesn't enter the opponent goal.
   Additionally a large punishment is given if the puck velocity is less than 5 m/s.
4. **Episode Termination**: If the puck trajectory doesn't enter the opponents goal (ignoring the opponent), then end the episode, otherwise continue the episode. In either approach the simulation doesn't reset.

**Benefits**:
- Dense reward signal for offensive play, along with frequent episode terminations making training easier.
- Velocity bonus encourages aggressive shots
- Continuing simulation (without reset) maintains diverse state distribution
- Prevents reward hacking where agent only cares to shoot frequently just to trigger halfway-line rewards
- By continuing the episode if the puck trajectory enters the goal, we still encourage unpredictable shots as they result in actual goals giving a large reward boost.

---

## Training Opponents

### Self-Play
The primary training method is **self-play**, where the agent plays against copies of itself. This creates a curriculum of increasing difficulty as the agent improves.

**Limitation**: Self-play alone results in overfitting to a single opponent strategy.

### Diverse Opponent Pool
To ensure robust performance, the agent trains against multiple opponent types:

#### 1. A Defender Agent
- Another RL agent trained to defend against opponent shots and stabailze the puck
- To make the defender agent possible to score on, we add a ~102 ms extra delay to the defenders vision data
- When puck stabalized on its side, we switch to the main policy for offensive play

#### 2. Algorithmic Agent
- Agent programmed to move side to side always blocking straight shots and encouraging bounce shots
- When the puck enters it's side, it switches to the main agent
- Mimics how humans often defend
- Each agent has a given fixed forward distance adding diversity

### 3. Previous Models
- Previously trained networks, for example networks that focus on bounce shots or straight shots

This opponent diversity prevents strategy overfitting and ensures the agent can handle various playing styles.

## Off Policy Learning

Since we are using Soft Actor-Critic (SAC), an off policy algorithm, to train the agent, we can use data gathered from another policy to help train the current policy. Becuase of this, all data collected by the previous models in the opponent pool are added to the data buffer for the main policy.

We noticed that if this wasn't done then the agent would specalize in one type of shot and forget all other shots. By adding two previous agents to the opponent pool, one that specalized in straight shots and another that specalized in bounce shots, we ensure that the final policy never forgets about different types of shots and specalized in both types.

# Hyperparameters

## Network Architecture

### Policy Network
A fully-connected deep neural network with **~200k parameters**:

```python
Layer 1:  Linear(obs_dim → 512) + LayerNorm + LeakyReLU
Layer 2:  Linear(512 → 256) + LayerNorm + LeakyReLU
Layer 3:  Linear(256 → 128) + LeakyReLU
Layer 4:  Linear(128 → 64) + LeakyReLU
Output:  Linear(64 → action_dim × 2) + NormalParamExtractor
```

**Design notes**:
- LayerNorm used in early layers for stable training with high-dimensional observations
- Output layer produces mean and standard deviation for a Gaussian policy

### Q Network
Similar architecture to the policy network. We used two Q networks and 2 target Q networks.

## Training Algorithm

**Soft Actor Critic (SAC)** with the following hyperparameters:

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| `lr_policy` | 5e-4 -> 1e-5 | Lower learning rate as it gets closer to convergence |
| `lr_value` | 5e-4 -> 1e-5  | Matched to policy learning rate |
| `gamma` | 0.98 | Low discount factor as air hockey is a fast game |
| `tau` | 0.001 | slow updating of target Q nets |
| `batch_size` | 1024 | Large batches enabled by fast simulation |

# Entropy

Although SAC does incentivise policies with higher entropy (hence the word soft in SAC here), we also tamper with the agents actions to sample surprising rollouts that could not be gotten with entropy alone.

Specifically, we modify the puck observation by applying a constant large shift to its observed position, sample the action, then remove the shift maintaining a valid transition pair. This produces useful rollouts where the agent learns multiple shot types are possible at many positions - further removing the chance of shot diversity collapse. This is only done to 30% of the agents with each agent getting a random offset.

# Domain Randomization

To ensure the trained policy transfers robustly to the physical system, we apply extensive domain randomization based on empirically measured noise characteristics.

## Puck Position Noise

### Dual-Layer Noise Model
Puck observations are corrupted with two noise sources:

1. **White Noise**: Gaussian noise with standard deviation measured from stationary puck position variance
2. **Perlin Noise**: Spatially correlated noise applied per-environment, with standard deviation based on calibration measurements of puck tracking accuracy vs. ground truth (see System Identification)

### Occlusion Handling
When the puck is partially occluded by support beams:

- **Partial occlusion** (>0% covered): Different white noise standard deviation, measured empirically
- **Full occlusion** (>50% covered): Probability of detection failure, in which case the puck buffer repeats the previous frame's data

**Note**: In reality, only 100% covered causes detection failure. However, since camera placement varies during setup (changing occlusion angles), this conservative approach ensures resilience to different configurations.

Occlusion zones are approximated based on camera positioning geometry.

## Mallet State Noise

- **Position noise**: White noise added to mallet position observations, standard deviation from measured sensor data
- **Velocity noise**: White noise added to calculated mallet velocity, based on empirical measurements

## Action Execution Noise

The feedforward + feedback control system does not perfectly track desired paths (see System Identification). To model this:

1. Add white noise to the agent's output `Vx` and `Vy` parameters
2. Standard deviations determined by:
   - Segmenting real vs. expected trajectories into small path segments
   - Optimizing each segment to find the `Vx`, `Vy` that best explain the actual motion
   - Computing standard deviation across all segments

This ensures the agent experiences realistic path-following errors during training.

## System Dynamics Randomization

### Feedforward Coefficients
Domain randomization is applied to the feedforward coefficients `(a₁, a₂, a₃, b₁, b₂, b₃)`. Critically:

- **Agent observes its own coefficients**: The current domain's coefficients are provided as part of the state
- **Opponent coefficients are hidden**: The agent does not know the opponent's dynamics

**Benefits**:
- **Adaptability**: If hardware changes shift system dynamics, new coefficients can be provided without retraining
- **Strategic diversity**: Agent faces faster and slower opponents, learning to counter different playstyles
- **Preparation**: Must plan for uncertainty in opponent capabilities

### Timing Delays
All timing delays are randomized according to measured distributions (see Firmware and Timing section):
- Mean delay
- Standard deviation
- Min/max bounds

This ensures the agent learns to act effectively despite information latency.

