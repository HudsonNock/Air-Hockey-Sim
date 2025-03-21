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
from collections import deque
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
import time

# Simulation parameters
obs_dim = 17
action_dim = 4
xbound = 2.0
ybound = 1.0
Vmax = 24
Ts = 0.1
mallet_r = 0.0509
puck_r = 0.0315
goal_width = 0.254
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

class ReplayBuffer:
    def __init__(self, buffer_size=1000000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        sampled_transitions = [self.buffer[i] for i in indices]
        return sampled_transitions

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


policy_module = TensorDictModule(
    policy_net, in_keys=["observation"], out_keys=["loc", "scale"]
)

policy_module = ProbabilisticActor(
    module=policy_module,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": torch.tensor([mallet_r, mallet_r, 0.3, 0.3]),
        "high": torch.tensor([xbound/2-mallet_r, ybound - mallet_r, 10, 10]),
    },
    default_interaction_type=tensordict.nn.InteractionType.RANDOM,
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
)

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

policy_module.load_state_dict(torch.load("policy_weights8.pth"))
value_module.load_state_dict(torch.load("value_weights8.pth"))

advantage_module = GAE(
    gamma=0.99, lmbda=0.5, value_network=value_module
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=0.01, #0.02
    entropy_bonus=True, 
    entropy_coef=0.00024 # 0.0003
    )

optim = torch.optim.Adam(loss_module.parameters(), lr=3e-4) #3e-4

envs = 2048
sim.initalize(envs=envs, mallet_r=mallet_r, puck_r=puck_r, goal_w=goal_width, V_max=Vmax)
mallet_init = np.array([[0.25, 0.5], [1.75, 0.5]])
mallet_pos = np.tile(mallet_init, (envs,1,1))
puck_init = np.array([[0.5, 0.5], [1.5, 0.5]])
obs = np.empty((2*envs, obs_dim))
attack = np.zeros((2*envs,), dtype=np.dtype(bool))
attack[:envs] = True
obs[:envs, :] = np.concatenate([mallet_init[0], mallet_init[1], puck_init[0], np.zeros((6,)), puck_init[0], np.zeros((2,)), np.array([1.0])])
obs[envs:, :] = np.concatenate([mallet_init[0], mallet_init[1], puck_init[1], np.zeros((6,)), puck_init[1], np.zeros((3,))])
obs = TensorDict({"observation": torch.tensor(obs, dtype=torch.float32)})
batch_size = 1024

left_puck = np.full((envs), True)
save_num = 0

theta = 0

if True:
    useDelay = False
    delay = 0.06

    buffer_size = 200_000
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(buffer_size)
    )

    #Simulation loop
    for update in range(500000):  # Training loop
        print("simulating...")
        for timestep in range(50):
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

            sim.impulse(0.0185, theta + np.random.uniform(-0.2,0.2))
            if (50*update + timestep) % 25 == 0:
                theta = np.random.uniform(0, 2*3.14)

            if useDelay:
                _, _, _, _, cross_left, cross_right = sim.step(delay)
                sim.take_action(xf, Vo)
                mallet_pos, puck_pos, mallet_vel, puck_vel, cross_left2, cross_right2 = sim.step(Ts-delay)
                cross_left = np.maximum(cross_left, cross_left2)
                cross_right = np.maximum(cross_right, cross_right2)
                puck_pos_noM, puck_vel_noM = sim.step_noM(Ts*2)
            else:
                sim.take_action(xf, Vo)
                mallet_pos, puck_pos, mallet_vel, puck_vel, cross_left, cross_right = sim.step(Ts)
                puck_pos_noM, puck_vel_noM = sim.step_noM(Ts*2)

            attack_copy = np.copy(attack)

            dones = np.full((2*envs), False)
            rewards = np.zeros((2*envs))

            #Shooting reward
            dones[:envs] = np.logical_or(cross_right > 0, cross_right == -0.5)
            dones[envs:] = np.logical_or(cross_left > 0, cross_left == -0.5)

            attack_copy[dones] = False

            rewards[:envs] += np.where(cross_right > 0,\
                                 np.where(attack[:envs], np.maximum(cross_right,0)/60, np.sqrt(np.maximum(cross_right,0))/120),\
                                 np.where(np.logical_and(cross_right == -0.5, np.logical_not(attack[:envs])), -0.3, 0))
            rewards[envs:] += np.where(cross_left > 0,\
                                 np.where(attack[envs:], np.maximum(cross_left, 0)/60, np.sqrt(np.maximum(cross_left, 0))/120),\
                                 np.where(np.logical_and(cross_left == -0.5, np.logical_not(attack[envs:])), -0.3, 0))

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
                    attack_copy[idx] = True
                    attack_copy[idx+envs] = False

            goals = sim.check_goal()
            for idx in range(envs):
                if goals[idx] == 1:
                    rewards[idx+envs] = -1

                elif goals[idx] == -1:
                    rewards[idx] = -1

                if goals[idx] != 0:
                    dones[idx] = True
                    dones[idx+envs] = True
                    mallet_pos[idx,:,:] = mallet_init
                    mallet_vel[idx,:,:] = 0.0
                    puck_pos[idx] = puck_init[0]
                    puck_vel[idx,:] = 0.0
                    puck_pos_noM[idx] = puck_init[0]
                    puck_vel_noM[idx,:] = 0.0
                    attack_copy[idx] = True
                    attack_copy[idx+envs] = False

            #Energy Penalty
            past_mallet = obs["observation"].numpy()[:, :2]
            past_mallet[envs:] = bounds - past_mallet[envs:]
            new_dist = np.sqrt(np.sum((past_mallet - np.concatenate([xf[:,0,:], xf[:,1,:]]))**2,axis=1))

            voltage = np.concatenate([actions_np[:envs, 2:], actions_np[envs:,2:]])
            voltage_cost = np.sqrt(np.sum(voltage**2, axis=1))

            rewards += -new_dist / 50 - voltage_cost / 1400

            #Near edge penalty
            #new_dist = np.max(np.abs(np.concatenate([puck_init[0] - xf[:,0,:], puck_init[1] - xf[:,1,:]])),axis=1)
            #rewards += np.where(new_dist > 0.44, -0.1, 0)

            #Stabalization
            dist_L = np.sqrt(np.sum((puck_pos - np.array([0.65,0.5]))**2, axis=1))
            dist_R = np.sqrt(np.sum((puck_pos - np.array([1.35, 0.5]))**2, axis=1))

            left_puck = (puck_pos[:,0] < xbound/2)
            #rewards[~attack_copy] -= 0.0003

            vel_norm = np.linalg.norm(puck_vel, axis=1)

            stabalized_l = ~attack_copy[:envs] & left_puck & (vel_norm > 0.7) & (np.abs(puck_vel[:,1])/np.maximum(np.abs(puck_vel[:,0]), 0.001) > 7) & (np.abs(puck_pos[:,0] - 0.65) < 0.3)
            stabalized_l = stabalized_l & np.random.choice([False, True], size=envs, p=[0.3, 0.7])
            stabalized_r = ~attack_copy[envs:] & ~left_puck & (vel_norm > 0.7) & (np.abs(puck_vel[:,1])/np.maximum(np.abs(puck_vel[:,0]), 0.001) > 7) & (np.abs(puck_pos[:,0] - 1.35) < 0.3)
            stabalized_r = stabalized_r & np.random.choice([False, True], size=envs, p=[0.3, 0.7])
            stabalized = np.concatenate([stabalized_l, stabalized_r])

            dones[stabalized] = True
            attack_copy[stabalized] = True

            rewards[:envs][stabalized_l] += 2.0
            rewards[envs:][stabalized_r] += 2.0

            #rewards -= 0.003

            attack = attack_copy

            next_obs_np = np.empty((2*envs, obs_dim))
            next_obs_np[:envs,:] = np.concatenate([mallet_pos[:,0,:], mallet_pos[:,1,:], puck_pos, mallet_vel[:,0,:], mallet_vel[:,1,:], puck_vel, puck_pos_noM, puck_vel_noM, attack[:envs].reshape(-1, 1)], axis=1) #(game, 12)
            next_obs_np[envs:,:] = np.concatenate([bounds - mallet_pos[:,1,:], bounds - mallet_pos[:,0,:], bounds - puck_pos, -mallet_vel[:,1,:], -mallet_vel[:,0,:], -puck_vel, bounds - puck_pos_noM, -puck_vel_noM, attack[envs:].reshape(-1,1)], axis=1)

            rewards = torch.tensor(rewards, dtype=torch.float32)  # Rewards
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32)  # Next observations
            terminated = torch.tensor(dones)
            dones = torch.tensor(dones) # Done flags (0 or 1)

            replay_buffer.extend(TensorDict({
                    "observation": obs["observation"],
                    "action": actions,
                    "reward": rewards,
                    "done": dones,
                    "next_observation": next_obs,
                    "terminated": terminated,
                    "sample_log_prob": log_prob
                }, batch_size=[actions.shape[0]]))

            obs = TensorDict({"observation": next_obs})

        print("training...")
        for _ in range(150):
            sample = replay_buffer.sample(batch_size)

            tensordict_data = TensorDict({
                "action": sample["action"],
                "observation": sample["observation"],
                # Create a nested structure for "next" fields
                "next": TensorDict({
                    "done": sample["done"],
                    "observation": sample["next_observation"],
                    "reward": sample["reward"],
                    "terminated": sample["terminated"]
                }, batch_size=[batch_size]),
                "sample_log_prob": sample["sample_log_prob"]
            }, batch_size=[batch_size])

            advantage_module(tensordict_data)

            loss_vals = loss_module(tensordict_data)
            loss_value = (
                    0.2 * loss_vals["loss_objective"]
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
            

            actions_np = sample["action"].numpy()
            actions_np[:,1] = ybound - actions_np[:,1]

            obs_np = sample["observation"].numpy()
            indicies_flip = [1,3,5,13]
            indicies_negate = [7,9,11,15]

            obs_np[:,indicies_flip] = ybound - obs_np[:,indicies_flip]
            obs_np[:,indicies_negate] *= -1

            next_obs_np = sample["next_observation"].numpy()
            next_obs_np[:,indicies_flip] = ybound - next_obs_np[:,indicies_flip]
            next_obs_np[:,indicies_negate] *= -1

            next_obs_np = torch.tensor(next_obs_np, dtype=torch.float32)

            tensordict_data = TensorDict({
                "action": torch.tensor(actions_np, dtype=torch.float32),
                "observation": torch.tensor(obs_np, dtype=torch.float32),
                # Create a nested structure for "next" fields
                "next": TensorDict({
                    "done": sample["done"],
                    "observation": next_obs_np,
                    "reward": sample["reward"],
                    "terminated": sample["terminated"]
                }, batch_size=[batch_size]),
                "sample_log_prob": sample["sample_log_prob"]
            }, batch_size=[batch_size])

            advantage_module(tensordict_data)

            loss_vals = loss_module(tensordict_data)
            loss_value = (
                     0.2 * loss_vals["loss_objective"]
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
            

        torch.save(policy_module.state_dict(), "policy_weights9.pth")
        torch.save(value_module.state_dict(), "value_weights9.pth")
        print((update+1) / 100)

    print("Done Training")

sim.initalize(envs=1, mallet_r=mallet_r, puck_r=puck_r, goal_w=goal_width, V_max=Vmax)
Ts = 0.1
envs = 1
N = 35
step_size = Ts/N
obs = np.empty((2, obs_dim))
attack = np.array([True, False])
obs[:1, :] = np.concatenate([mallet_init[0], mallet_init[1], puck_init[0], np.zeros((6,)), puck_init[0], np.zeros((2,)), np.array([1.0])])
obs[1:, :] = np.concatenate([mallet_init[0], mallet_init[1], puck_init[1], np.zeros((6,)), puck_init[1], np.zeros((3,))])
obs = TensorDict({"observation": torch.tensor(obs, dtype=torch.float32)}, batch_size = 2)
mallet_pos = np.tile(mallet_init, (1,1,1))
start_cnt = 0
theta = 0

left_puck = np.full((1), True)
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
    print(attack)
    #loc = policy_out["loc"].detach().numpy()
    #scale = policy_out["scale"].detach().numpy()
    #xf = np.stack([actions_np[:,:2][:1, :], actions_np[:,:2][1:,:]], axis=1)
    xf = np.stack([actions_np[:1, :2], actions_np[1:,:2]], axis=1)
    #Vo = np.stack([actions_np[:,2:][:1, :], actions_np[:,2:][1:,:]], axis=1)
    #Vo = np.full((1, 2, 2), 1)
    xf[:,1,:] = bounds - xf[:,1,:]
    Vo = np.stack([actions_np[:1, 2:], actions_np[1:,2:]], axis=1)

    sim.take_action(xf, Vo)
    sim.impulse(0.0185, theta + np.random.uniform(-0.2,0.2))
    if start_cnt % 30 == 0:
        theta = np.random.uniform(0, 2*3.14)
    start_cnt += 1
    #theta = theta + np.random.uniform(-0.1,0.3)

    #puck_wall_col = np.full((1,2), False)
    cross_left = np.full((1,), -1.0)
    cross_right = np.full((1,), -1.0)

    env_err = []
    goals = [0]
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
            break

        sim.display_state(0)
    
    puck_pos_noM, puck_vel_noM = sim.step_noM(Ts*2)

    attack_copy = np.copy(attack)

    dones = np.full((2*envs), False)
    rewards = np.zeros((2*envs))

    #Shooting reward
    dones[:envs] = np.logical_or(cross_right > 0, cross_right == -0.5)
    dones[envs:] = np.logical_or(cross_left > 0, cross_left == -0.5)

    attack_copy[dones] = False

    rewards[:envs] += np.where(cross_right > 0,\
                         np.where(attack[:envs], np.maximum(cross_right,0)/60, np.sqrt(np.maximum(cross_right,0))/120),\
                         np.where((cross_right == -0.5) and (np.logical_not(attack[:envs])), -0.3, 0))
    rewards[envs:] += np.where(cross_left > 0,\
                         np.where(attack[envs:], np.maximum(cross_left, 0)/60, np.sqrt(np.maximum(cross_left, 0))/120),\
                         np.where((cross_left == -0.5) and (np.logical_not(attack[envs:])), -0.3, 0))

    #print(rewards)
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

    for idx in range(envs):
        if goals[idx] == 1:
            rewards[idx+envs] = -1

        elif goals[idx] == -1:
            rewards[idx] = -1

        if goals[idx] != 0:
            dones[idx] = True
            dones[idx+envs] = True
            mallet_pos[idx,:,:] = mallet_init
            mallet_vel[idx,:,:] = 0.0
            puck_pos[idx] = puck_init[0]
            puck_vel[idx,:] = 0.0
            puck_pos_noM[idx] = puck_init[0]
            puck_vel_noM[idx,:] = 0.0
            attack_copy[idx] = True
            attack_copy[idx+envs] = False

    #Energy Penalty
    past_mallet = obs["observation"].numpy()[:, :2]
    past_mallet[envs:] = bounds - past_mallet[envs:]
    new_dist = np.sqrt(np.sum((past_mallet - np.concatenate([xf[:,0,:], xf[:,1,:]]))**2,axis=1))

    voltage = np.concatenate([actions_np[:envs, 2:], actions_np[envs:,2:]])
    voltage_cost = np.sqrt(np.sum(voltage**2, axis=1))

    rewards += -new_dist / 50 - voltage_cost / 1200

    #Near edge penalty
    new_dist = np.max(np.abs(np.concatenate([puck_init[0] - xf[:,0,:], puck_init[1] - xf[:,1,:]])),axis=1)
    rewards += np.where(new_dist > 0.44, -0.1, 0)

    #Stabalization
    dist_L = np.sqrt(np.sum((puck_pos - np.array([0.65,0.5]))**2, axis=1))
    dist_R = np.sqrt(np.sum((puck_pos - np.array([1.35, 0.5]))**2, axis=1))

    left_puck = (puck_pos[:,0] < xbound/2)
    rewards[~attack_copy] -= 0.001

    vel_norm = np.linalg.norm(puck_vel, axis=1)

    stabalized_l = ~attack_copy[:envs] & left_puck & (vel_norm > 0.7) & (np.abs(puck_vel[:,1])/np.maximum(np.abs(puck_vel[:,0]), 0.001) > 7) & (np.abs(puck_pos[:,0] - 0.65) < 0.3)
    stabalized_l = stabalized_l & np.random.choice([False, True], size=envs, p=[0.3, 0.7])
    stabalized_r = ~attack_copy[envs:] & ~left_puck & (vel_norm > 0.7) & (np.abs(puck_vel[:,1])/np.maximum(np.abs(puck_vel[:,0]), 0.001) > 7) & (np.abs(puck_pos[:,0] - 1.35) < 0.3)
    stabalized_r = stabalized_r & np.random.choice([False, True], size=envs, p=[0.3, 0.7])
    stabalized = np.concatenate([stabalized_l, stabalized_r])

    dones[stabalized] = True
    attack_copy[stabalized] = True

    rewards[:envs][stabalized_l] += 2.0
    rewards[envs:][stabalized_r] += 2.0

    attack = attack_copy
    #attack[:] = True

    #print(rewards)

    #print("---")
    #print(dones)
    #print(rewards)
    #print(attack)
    #time.sleep(1)

    next_obs = np.empty((2*envs, obs_dim))
    next_obs[:envs,:] = np.concatenate([mallet_pos[:,0,:], mallet_pos[:,1,:], puck_pos, mallet_vel[:,0,:], mallet_vel[:,1,:], puck_vel, puck_pos_noM, puck_vel_noM, attack[:envs].reshape(-1, 1)], axis=1) #(game, 12)
    #puck_pos += np.array([np.random.uniform(-0.004, 0.004), np.random.uniform(-0.004, 0.004)])
    next_obs[envs:,:] = np.concatenate([bounds - mallet_pos[:,1,:], bounds - mallet_pos[:,0,:], bounds - puck_pos, -mallet_vel[:,0,:], -mallet_vel[:,1,:], -puck_vel, bounds - puck_pos_noM, -puck_vel_noM, attack[envs:].reshape(-1,1)], axis=1)

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
