import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import Air_hockey_sim_vectorized as sim
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
import tensordict
from tensordict import TensorDict, TensorDictBase
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule
from torchrl.objectives.value import GAE
import time

# Simulation parameters
obs_dim = 17
action_dim = 4
xbound = 2.0
ybound = 1.0
Vmax = 24
Ts = 0.2
mallet_r = 0.05
puck_r = 0.05
bounds = np.array([xbound, ybound])

class ScaledNormalParamExtractor(NormalParamExtractor):
    def __init__(self, scale_factor=0.5):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        loc, scale = super().forward(x)
        #scale *= self.scale_factor
        scale = 2*self.scale_factor/ (1+torch.exp(-scale)) - self.scale_factor + 0.001
        return loc, scale

policy_net = nn.Sequential(
    nn.Linear(obs_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,32),
    nn.ReLU(),
    nn.Linear(32, action_dim * 2),
    ScaledNormalParamExtractor(),
)

#value_net = nn.Sequential(
#    nn.Linear(obs_dim, 64),
#    nn.ReLU(),
#    nn.Linear(64, 64),
#    nn.ReLU(),
#    nn.Linear(64, 64),
#    nn.ReLU(),
#    nn.Linear(64, 2*2),
#    ScaledNormalParamExtractor(),
#)

policy_module = TensorDictModule(
    policy_net, in_keys=["observation"], out_keys=["loc", "scale"]
)

policy_module = ProbabilisticActor(
    module=policy_module,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": torch.tensor([mallet_r, mallet_r, 1, 1]),
        "high": torch.tensor([xbound/2-mallet_r, ybound - mallet_r, 10, 10]),
    },
    default_interaction_type=tensordict.nn.InteractionType.RANDOM,
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
)

"""
policy_net2 = nn.Sequential(
    nn.Linear(obs_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,32),
    nn.ReLU(),
    nn.Linear(32, action_dim * 2),
    ScaledNormalParamExtractor(),
)

policy_module2 = TensorDictModule(
    policy_net2, in_keys=["observation"], out_keys=["loc", "scale"]
)

policy_module2 = ProbabilisticActor(
    module=policy_module2,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": torch.tensor([mallet_r, mallet_r, 1, 1]),
        "high": torch.tensor([xbound/2-mallet_r, ybound - mallet_r, 10, 10]),
    },
    default_interaction_type=tensordict.nn.InteractionType.RANDOM,
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
)

policy_module2.load_state_dict(torch.load("policy_module_weights7big.pth"))
"""

value_net = nn.Sequential(
    nn.Linear(obs_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,32),
    nn.ReLU(),
    nn.Linear(32, 1),
)

value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
    out_keys=["state_value"]
)

policy_module.load_state_dict(torch.load("policy_module_weights10big.pth"))
value_module.load_state_dict(torch.load("value_module_weights10big.pth"))

advantage_module = GAE(
    gamma=0.99, lmbda=0.95, value_network=value_module
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=0.05,
    entropy_bonus=True,
    entropy_coef=0.005
    )

optim = torch.optim.Adam(loss_module.parameters(), lr=1e-3)

envs = 2048
sim.initalize(envs=envs, mallet_r=0.05, puck_r=0.05, goal_w=0.35, V_max=Vmax)
mallet_init = np.array([[0.25, 0.5], [1.75, 0.5]])
mallet_pos = np.tile(mallet_init, (envs,1,1))
puck_init = np.array([[0.5, 0.5], [1.5, 0.5]])
obs = np.empty((2*envs, obs_dim))
obs[:envs, :] = np.concatenate([mallet_init[0], mallet_init[1], puck_init[0], np.zeros((7,)), puck_init[0], np.zeros((2,))])
obs[envs:, :] = np.concatenate([mallet_init[0], mallet_init[1], puck_init[1], np.zeros((7,)), puck_init[1], np.zeros((2,))])
obs = TensorDict({"observation": torch.tensor(obs, dtype=torch.float32)})

left_puck = np.full((envs), True)
timers = np.zeros((2*envs), dtype=np.float32)
save_num = 0

# Simulation loop
for update in range(500000):  # Training loop
    # Example: Simulated batch from your environment
    policy_out = policy_module(obs)
    #policy_out1 = policy_module(TensorDict({"observation": obs["observation"][envs:]}))
    #policy_out2 = policy_module2(TensorDict({"observation": obs["observation"][:envs]}))
    actions = policy_out["action"].detach()
    policy_out["sample_log_prob"] = torch.maximum(policy_out["sample_log_prob"], torch.tensor(-8, dtype=torch.float32))
    log_prob = policy_out["sample_log_prob"].detach()
    #actions = torch.concatenate([policy_out2["action"].detach(), policy_out1["action"].detach()])
    #log_prob = torch.concatenate([policy_out2["sample_log_prob"].detach(), policy_out1["sample_log_prob"].detach()])

    if np.isnan(log_prob.numpy()).any():
        print("NAN")
        break
    actions_np = actions.numpy()
    xf = np.stack([actions_np[:envs, :2], actions_np[envs:,:2]], axis=1)
    xf[:,1,:] = bounds - xf[:,1,:]
    Vo = np.stack([actions_np[:envs, 2:], actions_np[envs:,2:]], axis=1)
    sim.take_action(xf, Vo)

    mallet_pos, puck_pos, mallet_vel, puck_vel, cross_left, cross_right = sim.step(Ts)
    puck_pos_noM, puck_vel_noM = sim.step_noM(Ts)

    rewards = np.zeros((2*envs))
    #Shooting reward
    rewards[:envs] += np.where(cross_right > 0, np.sqrt(np.maximum(cross_right, 0)) / 2000 * timers[:envs], 0)
    rewards[envs:] += np.where(cross_left > 0, np.sqrt(np.maximum(cross_left, 0)) / 2000 * timers[envs:], 0)

    rewards[:envs] += np.where(cross_right == -0.5, -0.05, 0)
    rewards[envs:] += np.where(cross_left == -0.5, -0.05, 0)

    crossed = (cross_left != -1) | (cross_right != -1)
    timers[:envs][crossed] = 0
    timers[envs:][crossed] = 0
    left_puck = (puck_pos[:,0] < xbound/2)

    timers[:envs][np.logical_not(left_puck)] = 0
    timers[:envs][left_puck] += Ts
    timers[envs:][np.logical_not(left_puck)] += Ts
    timers[envs:][left_puck] = 0

    dones = np.full((2*envs), False)
    env_err = sim.check_state()

    if len(env_err) > 0:
        sim.reset_sim(env_err)
        for idx in env_err:
            mallet_pos[idx] = mallet_init
            puck_pos[idx] = puck_init[0]
            mallet_vel[idx,:,:] = 0.0
            puck_vel[idx,:] = 0.0
            puck_pos_noM[idx] = puck_init[0]
            puck_vel_noM[idx,:] = 0.0
            rewards[idx] = -1
            rewards[envs+idx] = -1
            dones[idx] = True
            dones[idx+envs] = True
            timers[idx] = 0
            timers[envs+idx] = 0
    
    goals = sim.check_goal()
    for idx in range(envs):
        if goals[idx] == 1:
            rewards[idx] = 1
            rewards[idx+envs] = -1
            
        elif goals[idx] == -1:
            rewards[idx] = -1
            rewards[idx+envs] = 1
        
        if goals[idx] != 0:
            dones[idx] = True
            dones[idx+envs] = True
            mallet_pos[idx,:,:] = mallet_init
            mallet_vel[idx,:,:] = 0.0
            puck_pos[idx] = puck_init[0]
            puck_vel[idx,:] = 0.0
            puck_pos_noM[idx] = puck_init[0]
            puck_vel_noM[idx,:] = 0.0
            timers[idx] = 0
            timers[envs+idx] = 0

    out_of_time = (timers > 10.05)
    if np.any(out_of_time):
        env_idx = np.concatenate([np.where(timers[:envs] > 10.05)[0], np.where(timers[envs:] > 10.05)[0]])
        sim.reset_sim(env_idx)
        for idx in env_idx:
            mallet_pos[idx] = mallet_init
            puck_pos[idx] = puck_init[0]
            mallet_vel[idx,:,:] = 0.0
            puck_vel[idx,:] = 0.0
            puck_pos_noM[idx] = puck_init[0]
            puck_vel_noM[idx,:] = 0.0
            dones[idx] = True
            dones[idx+envs] = True
            timers[idx] = 0
            timers[envs+idx] = 0

            if left_puck[idx]:
                rewards[idx] = -1.5
            else:
                rewards[idx+envs] = -1.5

    puck_v_norm = np.sqrt(np.sum(puck_vel**2, axis=1))

    #for idx in range(2*envs):
    #    rewards[idx] -= np.linalg.norm(past_actions[idx] - actions_np[idx]) * 0.05

    #Energy Penalty
    past_mallet = obs["observation"].numpy()[:, :2]
    past_mallet[envs:] = bounds - past_mallet[envs:]
    new_dist = np.sqrt(np.sum((past_mallet - np.concatenate([xf[:,0,:], xf[:,1,:]]))**2,axis=1))

    voltage = np.concatenate([actions_np[:envs, 2:], actions_np[envs:,2:]])
    voltage_cost = np.sqrt(np.sum(voltage**2, axis=1))

    rewards += -new_dist / 200 - voltage_cost / 1000

    #Near edge penalty
    new_dist = np.max(np.abs(np.concatenate([puck_init[0] - xf[:,0,:], puck_init[1] - xf[:,1,:]])),axis=1)
    rewards += np.where(new_dist > 0.44, -0.002, 0)

    #Possession reward
    dist_L = np.sqrt(np.sum((puck_pos - puck_init[0])**2, axis=1))
    dist_R = np.sqrt(np.sum((puck_pos - puck_init[1])**2, axis=1))

    left_puck = (puck_pos[:,0] < xbound/2)
    rewards[:envs][left_puck] += (10 - puck_v_norm[left_puck]) / 10000.0 + (0.4 - dist_L[left_puck]) / (400)
    rewards[envs:][~left_puck] += (10 - puck_v_norm[~left_puck]) / 10000.0 + (0.4 - dist_R[~left_puck]) / (400)

    next_obs = np.empty((2*envs, obs_dim))
    next_obs[:envs,:] = np.concatenate([mallet_pos[:,0,:], mallet_pos[:,1,:], puck_pos, mallet_vel[:,0,:], mallet_vel[:,1,:], puck_vel, np.tile(timers[:envs][:,np.newaxis], (1,1)), puck_pos_noM, puck_vel_noM], axis=1) #(game, 12)
    next_obs[envs:,:] = np.concatenate([bounds - mallet_pos[:,1,:], bounds - mallet_pos[:,0,:], bounds - puck_pos, -mallet_vel[:,0,:], -mallet_vel[:,1,:], -puck_vel, np.tile(timers[envs:][:,np.newaxis], (1,1)), bounds - puck_pos_noM, -puck_vel_noM], axis=1)

    rewards = torch.tensor(rewards, dtype=torch.float32)  # Rewards
    next_obs = torch.tensor(next_obs, dtype=torch.float32)  # Next observations
    terminated = torch.tensor(dones)
    dones = torch.tensor(dones) # Done flags (0 or 1)
    
    #tensordict_data = TensorDict({
    #    "action": actions[envs:],
    #    "observation": obs["observation"][envs:],
    #    "next": TensorDict({
    #        "done": dones[envs:],
    #        "observation": next_obs[envs:],
    #        "reward": rewards[envs:],
    #        "terminated": terminated[envs:]
    #    }, batch_size=[envs]),
    #    "sample_log_prob": log_prob[envs:]
    #}, batch_size=[envs])

    tensordict_data = TensorDict({
        "action": actions,
        "observation": obs["observation"],
        # Create a nested structure for "next" fields
        "next": TensorDict({
            "done": dones,
            "observation": next_obs,
            "reward": rewards,
            "terminated": terminated
        }, batch_size=[2*envs]),
        "sample_log_prob": log_prob
    }, batch_size=[2*envs])
   
    advantage_module(tensordict_data)

    loss_vals = loss_module(tensordict_data)
    loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

    # Perform optimization
    optim.zero_grad()
    #if np.isnan(loss_value.detach().numpy()):
    #    print("NAN")
    loss_value.backward()
    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1)
    optim.step()

    #for param in loss_module.parameters():
    #    if np.any(np.isnan(param.detach().numpy())):
    #        print("NAN")
    #    break

    obs = TensorDict({"observation": next_obs})

    if update % 100 == 99:
        torch.save(policy_module.state_dict(), "policy_module_weights10big.pth")
        torch.save(value_module.state_dict(), "value_module_weights10big.pth")
        print((update+1) / 100)

print("Done Training")

sim.initalize(envs=1, mallet_r=0.05, puck_r=0.05, goal_w=0.35, V_max=Vmax)
N = 100
step_size = Ts/N
obs = np.empty((2, obs_dim))
obs[:1, :] = np.concatenate([mallet_init[0], mallet_init[1], puck_init[0], np.zeros((7,)), puck_init[0], np.zeros((2,))])
obs[1:, :] = np.concatenate([mallet_init[0], mallet_init[1], puck_init[1], np.zeros((7,)), puck_init[1], np.zeros((2,))])
obs = TensorDict({"observation": torch.tensor(obs, dtype=torch.float32)}, batch_size = 2)
mallet_pos = np.tile(mallet_init, (1,1,1))
start_cnt = 0

left_puck = np.full((1), True)
timers = np.zeros((2,), dtype=np.float32)
while True:
    #choose a random action
    #policy_out1 = policy_module(TensorDict({"observation": obs["observation"][1:]}))
    #policy_out2 = policy_module2(TensorDict({"observation": obs["observation"][:1]}))
    #actions = torch.concatenate([policy_out2["action"].detach(), policy_out1["action"].detach()])
    #actions_np = actions.numpy()

    policy_out = policy_module(obs)
    actions_np = policy_out["action"].detach().numpy()

    values = value_module(obs)["state_value"].detach().numpy()
    print("----------")
    print(values)
    #loc = policy_out["loc"].detach().numpy()
    #scale = policy_out["scale"].detach().numpy()
    #xf = np.stack([actions_np[:,:2][:1, :], actions_np[:,:2][1:,:]], axis=1)
    xf = np.stack([actions_np[:1, :2], actions_np[1:,:2]], axis=1)
    #Vo = np.stack([actions_np[:,2:][:1, :], actions_np[:,2:][1:,:]], axis=1)
    #Vo = np.full((1, 2, 2), 1)
    xf[:,1,:] = bounds - xf[:,1,:]
    Vo = np.stack([actions_np[:1, 2:], actions_np[1:,2:]], axis=1)

    sim.take_action(xf, Vo)

    #puck_wall_col = np.full((1,2), False)
    cross_left = np.full((1,), -1.0)
    cross_right = np.full((1,), -1.0)

    env_err = []
    got_goal = False
    for i in range(N):
        mallet_pos, puck_pos, mallet_vel, puck_vel, cross_left_n, cross_right_n = sim.step(step_size)
        cross_left = np.maximum(cross_left, cross_left_n)
        cross_right = np.maximum(cross_right, cross_right_n)
        
        env_err = sim.check_state()
        if len(env_err) > 0:
            #sim.reset_sim(index)
            break

        goals = sim.check_goal() #returns 1 or -1 depending on what agent scored a goal
        if goals[0] != 0:
            got_goal = True
            break

        sim.display_state(0)
    
    puck_pos_noM, puck_vel_noM = sim.step_noM(Ts)

    rewards = np.zeros((2,))
    #Shooting reward
    rewards[:1] += np.where(cross_right > 0, np.sqrt(np.maximum(cross_right, 0)) / 1000 * timers[:1], 0)
    rewards[1:] += np.where(cross_left > 0, np.sqrt(np.maximum(cross_left, 0)) / 1000 * timers[1:], 0)

    rewards[:1] += np.where(cross_right == -0.5, -0.01, 0)
    rewards[1:] += np.where(cross_left == -0.5, -0.01, 0)

    #print("shooting reward")
    #print(rewards)
    rewards = np.zeros((2,))

    cross_left = np.maximum(cross_left, cross_left_n)
    cross_right = np.maximum(cross_right, cross_right_n)
    #puck_wall_col = puck_wall_col | puck_wc
    crossed = (cross_left != -1) | (cross_right != -1)
    timers[:1][crossed] = 0
    timers[1:][crossed] = 0
    left_puck = (puck_pos[:,0] < xbound/2)

    timers[:1][np.logical_not(left_puck)] = 0
    timers[:1][left_puck] += Ts
    timers[1:][np.logical_not(left_puck)] += Ts
    timers[1:][left_puck] = 0

    #env_err = sim.check_state()
    if len(env_err) > 0:
        sim.reset_sim(env_err)
        for idx in env_err:
            mallet_pos[idx] = mallet_init
            puck_pos[idx] = puck_init[0]
            puck_pos_noM[idx] = puck_init[0]
            mallet_vel[idx,:,:] = 0.0
            puck_vel[idx,:] = 0.0
            puck_vel_noM[idx,:] = 0.0
            timers[idx] = 0
            timers[1+idx] = 0
    
    if got_goal:
        for idx in range(1):
            mallet_pos[idx,:,:] = mallet_init
            mallet_vel[idx,:,:] = 0.0
            puck_vel[idx,:] = 0.0
            puck_vel_noM[idx,:] = 0.0
            puck_pos_noM[idx] = puck_init[0]
            puck_pos[idx] = puck_init[0]
            timers[idx] = 0
            timers[1+idx] = 0

    out_of_time = (timers > 10.05)
    if np.any(out_of_time):
        env_idx = np.concatenate([np.where(timers[:1] > 10.05)[0], np.where(timers[1:] > 10.05)[0]])
        sim.reset_sim(env_idx)
        for idx in env_idx:
            mallet_pos[idx] = mallet_init
            puck_pos[idx] = puck_init[0]
            puck_pos_noM[idx] = puck_init[0]
            puck_vel_noM[idx,:] = 0.0
            mallet_vel[idx,:,:] = 0.0
            puck_vel[idx,:] = 0.0
            timers[idx] = 0
            timers[1+idx] = 0

    puck_v_norm = np.sqrt(np.sum(puck_vel**2, axis=1))

    #for idx in range(2*envs):
    #    rewards[idx] -= np.linalg.norm(past_actions[idx] - actions_np[idx]) * 0.05

    #Energy Penalty
    past_mallet = obs["observation"].numpy()[:, :2]
    past_mallet[1:] = bounds - past_mallet[1:]
    new_dist = np.sqrt(np.sum((past_mallet - np.concatenate([xf[:,0,:], xf[:,1,:]]))**2,axis=1))

    voltage = np.concatenate([actions_np[:1, 2:], actions_np[1:,2:]])
    voltage_cost = np.sqrt(np.sum(voltage**2, axis=1))

    rewards += -new_dist / 2000 - voltage_cost / 30000

    #print("Energy penalty")
    #print(rewards)
    rewards = np.zeros((2,))

    #away from walls reward
    new_dist = np.sqrt(np.sum((np.concatenate([puck_init[0] - xf[:,0,:], puck_init[1] - xf[:,1,:]]))**2,axis=1))
    rewards += np.where(new_dist > 0.47, -0.002, 0)

    #Possession reward
    dist_L = np.sqrt(np.sum((puck_pos - puck_init[0])**2, axis=1))
    dist_R = np.sqrt(np.sum((puck_pos - puck_init[1])**2, axis=1))

    left_puck = (puck_pos[:,0] < xbound/2)
    rewards[:1][left_puck] += (4 - puck_v_norm[left_puck]) / 4000.0 + (0.4 - dist_L[left_puck]) / (400)
    rewards[1:][~left_puck] += (4 - puck_v_norm[~left_puck]) / 4000.0 + (0.4 - dist_R[~left_puck]) / (400)

    #print("possession reward")
    #print(rewards)

    next_obs = np.empty((2, obs_dim))
    next_obs[:1,:] = np.concatenate([mallet_pos[:,0,:], mallet_pos[:,1,:], puck_pos, mallet_vel[:,0,:], mallet_vel[:,1,:], puck_vel, np.tile(timers[:1][:,np.newaxis], (1,1)), puck_pos_noM, puck_vel_noM], axis=1) #(game, 12)
    next_obs[1:,:] = np.concatenate([bounds - mallet_pos[:,1,:], bounds - mallet_pos[:,0,:], bounds - puck_pos, -mallet_vel[:,0,:], -mallet_vel[:,1,:], -puck_vel, np.tile(timers[1:][:,np.newaxis], (1,1)), bounds - puck_pos_noM, -puck_vel_noM], axis=1)
    obs = TensorDict({"observation": torch.tensor(next_obs, dtype=torch.float32)}, batch_size=2)

    #TODO
    #total delay = information delay + NN delay + Ts = 0.06 + NN delay + Ts
    #have a master B(s) that decides when to predict a new action
    #policy pi(s) for that new action
    #HRL setup with 
    #problems: the sim wouldn't be useful since we would have a very small dt anyway
    #benifits: would lower the total delay for action predictions
    
    #Alternatives:
    #total delay is low enough so the model has time to react to fast shots
