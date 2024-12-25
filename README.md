# Air-Hockey-Sim

Currently a WIP

The goal of this repo is to train and deploy an agent to play air hockey. The air hockey table has been modified to have a coreXY belt configuration allowing the robot to move the mallet.

![air_hockey_cliponline-video-cutter com-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/45868e2b-58df-49db-9185-147e5af6fca6)
(low framerate due to GIF)

Completed:
* Coded and vectorized a air hockey simulation which runs based on a transfer function of the real world table dynamics (relating motor voltages to position) generated though collecting real world data. The simulation works by using numerical methods to solve equations of intersection between the puck and the mallet, as well as the puck and the wall.
* The robot outputs final position and curve hyperparameters, we then generate a path using the hyperparameters that converges to the final position and has a corresponding voltage max less than our supply voltage. This action space was chosen since it gives a lot of freedom to the agent, yet it is impossible for the agent to preform a sequence of actions resulting in a collision with the table walls. However, there is some degeneracy in the final position when the final position is sufficiently far away from its initial position so this action space may still be subject to change.
* Each agent is given its, the opponents, and the pucks position and velocity. Aswell as the time that has passed since the puck entered it's side of the table.
* Ran severel multiagent environments in parrallel and trained using PPO implemented by torchRL.
* Modified the reward function so the agent gets better at scoring even after defense was perfected. This is done by determining the trajectory of the puck in the next second after it crosses the center line, and if it's trajectory enters the opponents goal it receives a reward. Additionally, to speed up learning, a termination condition is met with negative reward if the puck stays on the agent's side for more than 5 seconds.

To Add:
* Noise and domain randomization (already added in the simulation, but currently training without it for a baseline result)
* Tune the reward structure and hyperparameters until the agent acts perfectly (potentially increase the model size). Modifications to the reward structure could be to incentivise the agent to keep the puck at a slow speed when it's on its side which would help perfect a goal scoring trajectory.
* Add a delay to the agents actions to better simulate the real world dynamics. Additionally, to minimize delay on the physical application, another action could be added that preservs the previos action and steps forward some small dt < Ts (sampling time of the agent). This would reduce the total reaction time by at most Ts, however, this setup would also take longer to train due to how the simulation was implemented.
