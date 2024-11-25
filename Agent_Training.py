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
    nn.Linear(32, 2 * 2),
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
        "low": torch.tensor([mallet_r, mallet_r]),
        "high": torch.tensor([xbound/2-mallet_r, ybound - mallet_r]),
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

policy_module.load_state_dict(torch.load("policy_module_weights5big.pth"))
value_module.load_state_dict(torch.load("value_module_weights5big.pth"))


advantage_module = GAE(
    gamma=0.95, lmbda=0.95, value_network=value_module
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=0.2,
    entropy_bonus=True,
    entropy_coef=0.02
)

optim = torch.optim.Adam(loss_module.parameters(), lr=5e-4)

envs = 1024
mini_batch_num = 512
batch_num = int(envs / mini_batch_num)
sim.initalize(envs=envs, mallet_r=0.05, puck_r=0.05, goal_w=0.35, V_max=Vmax)
mallet_init = np.array([[0.25, 0.5], [1.75, 0.5]])
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


# Simulation loop
for update in range(500000):  # Training loop
    # Example: Simulated batch from your environment
    policy_out = policy_module(obs)
    actions = policy_out["action"].detach()
    log_prob = policy_out["sample_log_prob"].detach()
    if np.isnan(log_prob.numpy()).any():
        print("NAN")
        break
    actions_np = actions.numpy()
    xf = np.stack([actions_np[:envs, :], actions_np[envs:,:]], axis=1)
    #Vo = np.stack([actions_np[:,2:][:envs, :], actions_np[:,2:][envs:,:]], axis=1)
    Vo = np.full((envs, 2, 2), 8)
    xf[:,1,:] = bounds - xf[:,1,:]
    sim.take_action(xf, Vo)

    mallet_pos, puck_pos, mallet_vel, puck_vel, mallet_hit = sim.step(Ts)
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
            rewards[idx] = -1
            rewards[envs+idx] = -1
            dones[idx] = True
            dones[idx+envs] = True
            timers[idx] = 0
            timers[envs+idx] = 0
    
    goals = sim.check_goal()
    for idx in range(envs):
        if goals[idx] == 1:
            puck_pos[idx] = puck_init[1]
            rewards[idx] = 1
            rewards[idx+envs] = -1
            
        elif goals[idx] == -1:
            puck_pos[idx] = puck_init[0]
            rewards[idx] = -1
            rewards[idx+envs] = 1
        
        if goals[idx] != 0:
            dones[idx] = True
            dones[idx+envs] = True
            mallet_pos[idx,:,:] = mallet_init
            mallet_vel[idx,:,:] = 0.0
            puck_vel[idx,:] = 0.0
            timers[idx] = 0
            timers[envs+idx] = 0

    out_of_time = (timers > 3.05)
    if np.any(out_of_time):
        env_idx = np.concatenate([np.where(timers[:envs] > 3.05)[0], np.where(timers[envs:] > 3.05)[0]])
        sim.reset_sim(env_idx)
        for idx in env_idx:
            mallet_pos[idx] = mallet_init
            puck_pos[idx] = puck_init[0]
            mallet_vel[idx,:,:] = 0.0
            puck_vel[idx,:] = 0.0
            dones[idx] = True
            dones[idx+envs] = True
            timers[idx] = 0
            timers[envs+idx] = 0

            if left_puck[idx]:
                rewards[idx] = -0.5
            else:
                rewards[idx+envs] = -0.5

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

    puck_pos_noM, puck_vel_noM = sim.step_noM(Ts)
    next_obs = np.empty((2*envs, obs_dim))
    next_obs[:envs,:] = np.concatenate([mallet_pos[:,0,:], mallet_pos[:,1,:], puck_pos, mallet_vel[:,0,:], mallet_vel[:,1,:], puck_vel, np.tile(timers[:envs][:,np.newaxis], (1,1)), puck_pos_noM, puck_vel_noM], axis=1) #(game, 12)
    next_obs[envs:,:] = np.concatenate([bounds - mallet_pos[:,1,:], bounds - mallet_pos[:,0,:], bounds - puck_pos, -mallet_vel[:,0,:], -mallet_vel[:,1,:], -puck_vel, np.tile(timers[envs:][:,np.newaxis], (1,1)), bounds - puck_pos_noM, -puck_vel_noM], axis=1)

    rewards = torch.tensor(rewards, dtype=torch.float32)  # Rewards
    next_obs = torch.tensor(next_obs, dtype=torch.float32)  # Next observations
    dones = torch.tensor(dones) # Done flags (0 or 1)
    terminated = torch.zeros(2*envs, dtype=torch.bool)

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

    for batch in range(batch_num):
        mini_batch = tensordict_data[(batch*mini_batch_num):((batch+1)*mini_batch_num)]
        loss_vals = loss_module(mini_batch)
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

    #loss_vals = loss_module(tensordict_data)
    #loss_value = (
    #            loss_vals["loss_objective"]
    #            + loss_vals["loss_critic"]
    #            #+ loss_vals["loss_entropy"]
    #        )

    # Perform optimization
    #optim.zero_grad()
    #loss_value.backward()
    #torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
    #optim.step()

    obs = TensorDict({"observation": next_obs})

    if update % 100 == 99:
        torch.save(policy_module.state_dict(), "policy_module_weights5big.pth")
        torch.save(value_module.state_dict(), "value_module_weights5big.pth")
        print((update+1) / 100)
    #print(loss_value)
    #print(f"Update {update + 1}: Loss = {loss.item()}")

    
print("Done Training")
sim.initalize(envs=1, mallet_r=0.05, puck_r=0.05, goal_w=0.35, V_max=Vmax)
N = 200
step_size = Ts/N
obs = np.empty((2, obs_dim))
obs[:1, :] = np.concatenate([mallet_init[0], mallet_init[1], puck_init[0], np.zeros((7,)), puck_init[0], np.zeros((2,))])
obs[1:, :] = np.concatenate([mallet_init[0], mallet_init[1], puck_init[1], np.zeros((7,)), puck_init[1], np.zeros((2,))])
obs = TensorDict({"observation": torch.tensor(obs, dtype=torch.float32)})

left_puck = np.full((1), True)
timers = np.zeros((2,), dtype=np.float32)
while True:
    #choose a random action
    policy_out = policy_module(obs)
    actions_np = policy_out["action"].detach().numpy()
    values = value_module(obs)["state_value"].detach().numpy()
    print(values)
    #loc = policy_out["loc"].detach().numpy()
    #scale = policy_out["scale"].detach().numpy()
    #xf = np.stack([actions_np[:,:2][:1, :], actions_np[:,:2][1:,:]], axis=1)
    xf = np.stack([actions_np[:1, :], actions_np[1:,:]], axis=1)
    #Vo = np.stack([actions_np[:,2:][:1, :], actions_np[:,2:][1:,:]], axis=1)
    Vo = np.full((1, 2, 2), 8)
    xf[:,1,:] = bounds - xf[:,1,:]
    #xf[:,1,:] = np.array([(xbound/2 - 2*mallet_r)*np.random.rand() + xbound/2+mallet_r, mallet_r + (ybound-2*mallet_r)*np.random.rand()])
    sim.take_action(xf, Vo)

    for i in range(N-1):
        sim.step(step_size)
        
        index = sim.check_state()
        if len(index) > 0:
            sim.reset_sim(index)

        goals = sim.check_goal() #returns 1 or -1 depending on what agent scored a goal
        for idx in range(1):
            if goals[idx] != 0:
                timers[idx] = 0
                timers[1+idx] = 0
        sim.display_state(0)

    mallet_pos, puck_pos, mallet_vel, puck_vel, _ = sim.step(step_size)
    left_puck = (puck_pos[:,0] < xbound/2)

    timers[:1][np.logical_not(left_puck)] = 0
    timers[:1][left_puck] += Ts
    timers[1:][np.logical_not(left_puck)] += Ts
    timers[1:][left_puck] = 0

    env_err = sim.check_state()
    if len(env_err) > 0:
        sim.reset_sim(env_err)
        for idx in env_err:
            mallet_pos[idx] = mallet_init
            puck_pos[idx] = puck_init[0]
            mallet_vel[idx,:,:] = 0.0
            puck_vel[idx,:] = 0.0
            timers[idx] = 0
            timers[1+idx] = 0
    
    goals = sim.check_goal()
    for idx in range(1):
        if goals[idx] == 1:
            puck_pos[idx] = puck_init[1]
            
        elif goals[idx] == -1:
            puck_pos[idx] = puck_init[0]
        
        if goals[idx] != 0:
            mallet_pos[idx,:,:] = mallet_init
            mallet_vel[idx,:,:] = 0.0
            puck_vel[idx,:] = 0.0
            timers[idx] = 0
            timers[1+idx] = 0

    out_of_time = (timers > 3.05)
    if np.any(out_of_time):
        env_idx = np.concatenate([np.where(timers[:1] > 3.05)[0], np.where(timers[1:] > 3.05)[0]])
        sim.reset_sim(env_idx)
        for idx in env_idx:
            mallet_pos[idx] = mallet_init
            puck_pos[idx] = puck_init[0]
            mallet_vel[idx,:,:] = 0.0
            puck_vel[idx,:] = 0.0
            timers[idx] = 0
            timers[1+idx] = 0

    sim.display_state(0)
    #time.sleep(1)
    puck_pos_noM, puck_vel_noM = sim.step_noM(Ts)

    next_obs = np.empty((2, obs_dim))
    next_obs[:1,:] = np.concatenate([mallet_pos[:,0,:], mallet_pos[:,1,:], puck_pos, mallet_vel[:,0,:], mallet_vel[:,1,:], puck_vel, np.tile(timers[:1][:,np.newaxis], (1,1)), puck_pos_noM, puck_vel_noM], axis=1) #(game, 12)
    next_obs[1:,:] = np.concatenate([bounds - mallet_pos[:,1,:], bounds - mallet_pos[:,0,:], bounds - puck_pos, -mallet_vel[:,0,:], -mallet_vel[:,1,:], -puck_vel, np.tile(timers[1:][:,np.newaxis], (1,1)), bounds - puck_pos_noM, -puck_vel_noM], axis=1)
    obs = TensorDict({"observation": torch.tensor(next_obs, dtype=torch.float32)})
