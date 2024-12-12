# Air-Hockey-Sim

Currently a WIP

![air_hockey_cliponline-video-cutter com-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/45868e2b-58df-49db-9185-147e5af6fca6)

Completed:
* Coded and vectorized a air hockey simulation which runs based on a transfer function of the real world table dynamics generated though collecting real world data. The simulation works by using numerical methods to solve equations of intersection between the puck and the mallet, as well as the puck and the wall.
* Ran severel multiagent environments in parrallel and trained using PPO implemented by torchRL
* Modified the reward function so the agent gets better at scoring even after defense was perfected. This is done by determining the trajectory of the puck in the next second after it crosses the center line, and if it's trajectory enters the opponents goal it receives a reward.

To Add:
* Noise and domain randomization (already added in the simulation, but currently training without it for a baseline result)
* Tune the reward structure and hyperparameters until the agent acts perfectly (potentially increase the model size)
* Add a delay to the agents actions to better simulate the real world dynamics
