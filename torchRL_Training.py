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

#value_net = nn.Sequential(
#    nn.Linear(obs_dim, 64),
#    nn.ReLU(),
#    nn.Linear(64, 64),
#    nn.ReLU(),
#    nn.Linear(64, 64),
#    nn.ReLU(),
#    nn.Linear(64, 1),
#)

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
    clip_epsilon=0.01,
    entropy_bonus=True,
    entropy_coef=0.005
    )

optim = torch.optim.Adam(loss_module.parameters(), lr=3e-4)

envs = 2048
mini_batch_num = 512
batch_num = int(envs / mini_batch_num)
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
#past_actions = np.empty((2*envs, 2))
#past_actions[:envs, :] = mallet_init[0]
#past_actions[envs:,:] = mallet_init[1]
save_num = 0

# Simulation loop
for update in range(500000):  # Training loop
    # Example: Simulated batch from your environment
    policy_out = policy_module(obs)
    #values = value_module(obs)["state_value"].detach().numpy()
    #print(values[0])
    actions = policy_out["action"].detach()
    log_prob = policy_out["sample_log_prob"].detach()
    if np.isnan(log_prob.numpy()).any():
        print("NAN")
        break
    actions_np = actions.numpy()
    xf = np.stack([actions_np[:envs, :2], actions_np[envs:,:2]], axis=1)
    xf[:,1,:] = bounds - xf[:,1,:]
    #Vo = np.stack([actions_np[:,2:][:envs, :], actions_np[:,2:][envs:,:]], axis=1)
    #r = xf - mallet_pos
    #r_mag = np.linalg.norm(r, axis=-1)
    #Vo = np.abs(r/np.tile(r_mag[:,:,np.newaxis], (1,1,2))) * 13 + 1
    #Vo = np.full((envs, 2, 2), 8)
    Vo = np.stack([actions_np[:envs, 2:], actions_np[envs:,2:]], axis=1)
    sim.take_action(xf, Vo)

    mallet_pos, puck_pos, mallet_vel, puck_vel, cross_left, cross_right = sim.step(Ts)
    puck_pos_noM, puck_vel_noM = sim.step_noM(Ts)

    crossed = (cross_left != -1) | (cross_right != -1)
    timers[:envs][crossed] = 0
    timers[envs:][crossed] = 0
    left_puck = (puck_pos[:,0] < xbound/2)

    timers[:envs][np.logical_not(left_puck)] = 0
    timers[:envs][left_puck] += Ts
    timers[envs:][np.logical_not(left_puck)] += Ts
    timers[envs:][left_puck] = 0

    rewards = np.zeros((2*envs))
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

    out_of_time = (timers > 5.05)
    if np.any(out_of_time):
        env_idx = np.concatenate([np.where(timers[:envs] > 5.05)[0], np.where(timers[envs:] > 5.05)[0]])
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
                rewards[idx] = -1.0
            else:
                rewards[idx+envs] = -1.0

    #puck_pos_noM, puck_vel_noM = sim.step_noM(0.3)
    #left_goal = (puck_pos_noM[:,0] < 0)# & (puck_pos[:,0] > xbound/2)
    #right_goal = (puck_pos_noM[:,0] > xbound)# & (puck_pos[:,0] < xbound/2)
    #slope = np.abs(puck_vel_noM[:,1] / np.where(puck_vel_noM[:,0] == 0, 0.001, puck_vel_noM[:,0]))

    #rewards[np.concatenate([right_goal, np.full_like(right_goal, False)])] += 0.001 * np.sum(puck_vel[right_goal]**2, axis=1) * np.where(slope[right_goal] > 0.3, 1.34, 1)
    #rewards[np.concatenate([np.full_like(left_goal, False), left_goal])] += 0.001 * np.sum(puck_vel[left_goal]**2, axis=1) * np.where(slope[left_goal] > 0.3, 1.34, 1)

    #rewards[np.concatenate([puck_wall_collision[:,1], puck_wall_collision[:,0]])] -= 0.05

    #for idx in range(envs):
    #    if mallet_hit[idx,0]:
    #        rewards[idx] += 0.1
    #    elif mallet_hit[idx,1]:
    #        rewards[idx+envs] += 0.1

    #for idx in range(2*envs):
    #    rewards[idx] -= np.linalg.norm(past_actions[idx] - actions_np[idx]) * 0.05

    #for idx in range(envs):
    #    if mallet_pos[idx,0,0] > puck_pos[idx,0]:
    #        rewards[idx] -= 0.02
    #    if mallet_pos[idx,1,0] < puck_pos[idx,0]:
    #        rewards[idx+envs] -= 0.02

    #past_actions = actions_np
    #np.maximum(0.001 * cross_right, 0.05)
    rewards[:envs] += np.where(cross_right > 0, 0.001 * cross_right, np.where(cross_right==-0.5, -0.1,0))
    rewards[envs:] += np.where(cross_left > 0, 0.001 * cross_left, np.where(cross_left==-0.5, -0.1,0))

    #dones[:envs] = dones[:envs] | (cross_right==-0.5)
    #dones[envs:] = dones[envs:] | (cross_left==-0.5)


    #for idx in range(envs):
    #    if puck_pos[idx,0] < xbound/2:
    #        rewards[idx] += 0.002
    #        rewards[idx+envs] -= 0.002
    #    else:
    #        rewards[idx+envs] += 0.002
    #        rewards[idx] -= 0.002

    next_obs = np.empty((2*envs, obs_dim))
    next_obs[:envs,:] = np.concatenate([mallet_pos[:,0,:], mallet_pos[:,1,:], puck_pos, mallet_vel[:,0,:], mallet_vel[:,1,:], puck_vel, np.tile(timers[:envs][:,np.newaxis], (1,1)), puck_pos_noM, puck_vel_noM], axis=1) #(game, 12)
    next_obs[envs:,:] = np.concatenate([bounds - mallet_pos[:,1,:], bounds - mallet_pos[:,0,:], bounds - puck_pos, -mallet_vel[:,0,:], -mallet_vel[:,1,:], -puck_vel, np.tile(timers[envs:][:,np.newaxis], (1,1)), bounds - puck_pos_noM, -puck_vel_noM], axis=1)

    rewards = torch.tensor(rewards, dtype=torch.float32)  # Rewards
    next_obs = torch.tensor(next_obs, dtype=torch.float32)  # Next observations
    terminated = torch.tensor(dones)
    dones = torch.tensor(dones) # Done flags (0 or 1)

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
   # terminated = torch.zeros(1, 10, 1, dtype=torch.bool
    #        >>> advantage, value_target = module(obs=obs, next_reward=reward, next_done=done, next_obs=next_obs, next_terminated=terminated)
    advantage_module(tensordict_data)

    #for batch in range(batch_num):
    #    mini_batch = tensordict_data[(batch*mini_batch_num):((batch+1)*mini_batch_num)]
    #    loss_vals = loss_module(mini_batch)
    #    loss_value = (
    #                 loss_vals["loss_objective"]
    #                + loss_vals["loss_critic"]
    #                + loss_vals["loss_entropy"]
    #            )

        # Perform optimization
    #    optim.zero_grad()
    #    loss_value.backward()
     #   torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
     #   optim.step()

    loss_vals = loss_module(tensordict_data)
    loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

    # Perform optimization
    optim.zero_grad()
    loss_value.backward()
    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
    optim.step()

    obs = TensorDict({"observation": next_obs})

    if update % 100 == 99:
        torch.save(policy_module.state_dict(), "policy_module_weights11big.pth")
        torch.save(value_module.state_dict(), "value_module_weights11big.pth")
        print((update+1) / 100)

    #if update % 5000 == 4999:
    #    save_num += 1
    #    torch.save(policy_module.state_dict(), f"policy_module_weights_overnight_{save_num}.pth")
    #    torch.save(value_module.state_dict(), f"value_module_weights_overnight_{save_num}.pth")
    #print(loss_value)
    #print(f"Update {update + 1}: Loss = {loss.item()}")


print("Done Training")
"""
policy_module2 = TensorDictModule(
    policy_net, in_keys=["observation"], out_keys=["loc", "scale"]
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

policy_module2.load_state_dict(torch.load("policy_module_weights8big.pth"))
"""
sim.initalize(envs=1, mallet_r=0.05, puck_r=0.05, goal_w=0.35, V_max=Vmax)
N = 200
step_size = Ts/N
obs = np.empty((2, obs_dim))
obs[:1, :] = np.concatenate([mallet_init[0], mallet_init[1], puck_init[0], np.zeros((7,)), puck_init[0], np.zeros((2,))])
obs[1:, :] = np.concatenate([mallet_init[0], mallet_init[1], puck_init[1], np.zeros((7,)), puck_init[1], np.zeros((2,))])
obs = TensorDict({"observation": torch.tensor(obs, dtype=torch.float32)}, batch_size = 2)
mallet_pos = np.tile(mallet_init, (1,1,1))

left_puck = np.full((1), True)
timers = np.zeros((2,), dtype=np.float32)
while True:
    #choose a random action
    #policy_out1 = policy_module(obs[0:1])
    #policy_out2 = policy_module2(obs[1:2])
    #actions_np = np.concatenate([policy_out1["action"].detach().numpy(), policy_out2["action"].detach().numpy()])
    policy_out = policy_module(obs)
    actions_np = policy_out["action"].detach().numpy()

    values = value_module(obs)["state_value"].detach().numpy()
    print(values)
    #loc = policy_out["loc"].detach().numpy()
    #scale = policy_out["scale"].detach().numpy()
    #xf = np.stack([actions_np[:,:2][:1, :], actions_np[:,:2][1:,:]], axis=1)
    xf = np.stack([actions_np[:1, :2], actions_np[1:,:2]], axis=1)
    #Vo = np.stack([actions_np[:,2:][:1, :], actions_np[:,2:][1:,:]], axis=1)
    #Vo = np.full((1, 2, 2), 1)
    xf[:,1,:] = bounds - xf[:,1,:]
    #r = xf - mallet_pos
    #r_mag = np.linalg.norm(r, axis=-1)
    #Vo = np.abs(r * 6 * np.tile((np.log(r_mag + 6) / r_mag)[:,:,np.newaxis], (1,1,2))) + 1
    #Vo = np.abs(r/np.tile(r_mag[:,:,np.newaxis], (1,1,2))) * 13 + 1
    Vo = np.stack([actions_np[:1, 2:], actions_np[1:,2:]], axis=1)
    #xf[:,1,:] = np.array([(xbound/2 - 2*mallet_r)*np.random.rand() + xbound/2+mallet_r, mallet_r + (ybound-2*mallet_r)*np.random.rand()])
    sim.take_action(xf, Vo)
    #puck_wall_col = np.full((1,2), False)
    cross_left = np.full((1,), -1.0)
    cross_right = np.full((1,), -1.0)

    env_err = []
    got_goal = False
    for i in range(N):
        mallet_pos, puck_pos, mallet_vel, puck_vel, cross_left_n, cross_right_n = sim.step(step_size)
        #puck_wall_col = puck_wall_col | puck_wc
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
        #for idx in range(1):
        #    if goals[idx] != 0:

        #        timers[idx] = 0
        #        timers[1+idx] = 0
        #        got_goal = True
        sim.display_state(0)

    
    puck_pos_noM, puck_vel_noM = sim.step_noM(Ts)

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

    out_of_time = (timers > 5.05)
    if np.any(out_of_time):
        env_idx = np.concatenate([np.where(timers[:1] > 5.05)[0], np.where(timers[1:] > 5.05)[0]])
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

    sim.display_state(0)
    #print(cross_left)
    #print(cross_right)
    #time.sleep(1)

    #puck_pos_noM, puck_vel_noM = sim.step_noM(0.3)
    #left_goal = (puck_pos_noM[:,0] < 0)# & (puck_pos[:,0] > xbound/2)
    #right_goal = (puck_pos_noM[:,0] > xbound)# & (puck_pos[:,0] < xbound/2)
    #slope = np.abs(puck_vel_noM[:,1] / np.where(puck_vel_noM[:,0] == 0, 0.001, puck_vel_noM[:,0]))

    rewards = np.zeros((2,))
    #rewards[np.concatenate([right_goal, np.full_like(right_goal, False)])] += 0.001 * np.sum(puck_vel[right_goal]**2, axis=1) * np.where(slope[right_goal] > 0.3, 1.37, 1)
    #rewards[np.concatenate([np.full_like(left_goal, False), left_goal])] += 0.001 * np.sum(puck_vel[left_goal]**2, axis=1) * np.where(slope[left_goal]>0.3, 1.37, 1)
    #rewards[np.concatenate([puck_wall_col[:,1], puck_wall_col[:,0]])] -= 0.01

    rewards[:1] += np.where(cross_right > 0, 0.001 * cross_right, np.where(cross_right==-0.5, -0.04,0))
    rewards[1:] += np.where(cross_left > 0, 0.001 * cross_left, np.where(cross_left==-0.5, -0.04,0))

    #print(rewards)


    next_obs = np.empty((2, obs_dim))
    next_obs[:1,:] = np.concatenate([mallet_pos[:,0,:], mallet_pos[:,1,:], puck_pos, mallet_vel[:,0,:], mallet_vel[:,1,:], puck_vel, np.tile(timers[:1][:,np.newaxis], (1,1)), puck_pos_noM, puck_vel_noM], axis=1) #(game, 12)
    next_obs[1:,:] = np.concatenate([bounds - mallet_pos[:,1,:], bounds - mallet_pos[:,0,:], bounds - puck_pos, -mallet_vel[:,0,:], -mallet_vel[:,1,:], -puck_vel, np.tile(timers[1:][:,np.newaxis], (1,1)), bounds - puck_pos_noM, -puck_vel_noM], axis=1)
    obs = TensorDict({"observation": torch.tensor(next_obs, dtype=torch.float32)}, batch_size=2)
    
