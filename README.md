# AI Air Hockey

# Abstract

Our goal is to deploy a neural network based agent to play air hockey against a human at a professional level. Although we do not have access to a professional air hockey player, the current version already outpreforms all humans that have played against it. To acheive this level of preformance, we designed and statistically modeled the real preformance of the table, including the vision accuracry, motor responses to voltage, timing throughout the firmware, and puck dynamics - while adjusting the electromechanical system so our models become more accurate. Using this model we wrote our own vectorized simulation which runs at 450x real time speed to train a reinforcement learning (RL) agent which was then deployed on the physcial air hockey table.

The first half of the project (8 month period) and its full in depth report is linked below. This covers the simulation design, vision calibrations, and reinforcement learning setup. However, note that much of the content in this final report has since been improved, including the reinforcement learning state space and reward function, as well as puck collision dynamics in the simulation.
[First Half Final Report](docs/report.pdf)

![air_hockey_cliponline-video-cutter com-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/45868e2b-58df-49db-9185-147e5af6fca6)
(low framerate due to GIF)

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
