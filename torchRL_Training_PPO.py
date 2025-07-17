import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import Air_hockey_sim_vectorized as sim
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss, SACLoss
import tensordict
from tensordict import TensorDict, TensorDictBase
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule
from torchrl.objectives.value import GAE
from collections import deque
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
import time
from perlin_numpy import generate_perlin_noise_2d
from torchrl.data.tensor_specs import BoundedTensorSpec, CompositeSpec
import copy

load_filepath = "checkpoints/model_02.pth"
save_filepath = "checkpoints/model_02.pth"
train = False
# Simulation parameters
obs_dim = 52 #[puck_pos, opp_mallet_pos] #guess mean low stdl
            #[mallet_pos, mallet_vel, past action, past mallet_pos]
            #[e_n0, e_nr, e_nf, e_t0, e_tr, e_tf, var] x2 mallet and wall,
            #[a1, a2, a3, b1, b2]
            #[time]

mallet_r = 0.05082
puck_r = 0.5 * 0.0618
pullyR = 0.035306

#col: n_f + (1-n_f/n_0) * 2/(1+e^(n_r x^2)) n_0

action_dim = 4
Vmax = 24*0.8

sensor_error = 0.0027 #mean of 1mm between [0-2.5 mm]

#mean, std, min, max
image_delay = [15.166/1000, 0.3/1000, 14/1000, 16.5/1000]
mallet_delay = [7.17/1000, 0.3/1000, 6.2/1000, 8.0/1000]
camera_period = 1/120.0

frames = [0, 1, 2, 5, 11]

height = 1.9885
width = 0.9905
bounds = np.array([height, width])
goal_width = 0.254

entropy_coeff = 0.00024
epsilon = 0.05
gamma = 1.0
lmbda = 0.5
lr_policy = 5e-4
lr_value = 5e-4
batch_size = 2048
horizon = 60*1

class ScaledNormalParamExtractor(NormalParamExtractor):
    def __init__(self, scale_factor=0.8):
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
    
scale_factor = 0.5
if not train:
    scale_factor = 0.8
"""
class PolicyDense(nn.Module):
    def __init__(self, obs_dim, action_dim, scale_factor=0.8):
        super().__init__()
        self.scale_factor = scale_factor
        self.input_layer = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        self.dense1 = nn.Sequential(
            nn.Linear(obs_dim + 256, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        self.dense2 = nn.Sequential(
            nn.Linear(obs_dim + 256 * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        self.dense3 = nn.Sequential(
            nn.Linear(obs_dim + 256 * 3, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        self.output_layer = nn.Linear(obs_dim + 256 * 3 + 128, action_dim * 2)
        self.param_extractor = ScaledNormalParamExtractor(scale_factor=scale_factor)

    def forward(self, obs):
        x0 = self.input_layer(obs)                    # -> (batch, 256)
        x1 = self.dense1(torch.cat([obs, x0], dim=-1))  # -> (batch, 256)
        x2 = self.dense2(torch.cat([obs, x0, x1], dim=-1))  # -> (batch, 256)
        x3 = self.dense3(torch.cat([obs, x0, x1, x2], dim=-1))  # -> (batch, 128)

        x_all = torch.cat([obs, x0, x1, x2, x3], dim=-1)  # Final concat
        x_out = self.output_layer(x_all)

        return self.param_extractor(x_out)

class ValueDense(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.scale_factor = scale_factor
        self.input_layer = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        self.dense1 = nn.Sequential(
            nn.Linear(obs_dim + 256, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        self.dense2 = nn.Sequential(
            nn.Linear(obs_dim + 256 * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        self.dense3 = nn.Sequential(
            nn.Linear(obs_dim + 256 * 3, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        self.output_layer = nn.Linear(obs_dim + 256 * 3 + 128, 1)

    def forward(self, obs):
        x0 = self.input_layer(obs)                    # -> (batch, 256)
        x1 = self.dense1(torch.cat([obs, x0], dim=-1))  # -> (batch, 256)
        x2 = self.dense2(torch.cat([obs, x0, x1], dim=-1))  # -> (batch, 256)
        x3 = self.dense3(torch.cat([obs, x0, x1, x2], dim=-1))  # -> (batch, 128)

        x_all = torch.cat([obs, x0, x1, x2, x3], dim=-1)  # Final concat
        x_out = self.output_layer(x_all)

        return x_out
"""

policy_net = nn.Sequential(
    nn.Linear(obs_dim, 1024),
    nn.LayerNorm(1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.LayerNorm(1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.LayerNorm(512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, action_dim * 2),
    ScaledNormalParamExtractor(scale_factor=scale_factor),
)
#policy_net = PolicyDense(obs_dim, action_dim, scale_factor)

action_center = torch.tensor([0, 0, 2, 0], dtype=torch.float32)

def init_policy(m):
    if isinstance(m, nn.Linear):
        if m.out_features == action_dim * 2:
            # Initialize weights to 0 so output = bias
            nn.init.constant_(m.weight, 0.0)
            # Bias: loc = center, log_std = 0
            bias = torch.cat([action_center, torch.ones(action_dim)*2], dim=0)
            with torch.no_grad():
                m.bias.copy_(bias)
        else:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

policy_net.apply(init_policy)

policy_module = TensorDictModule(
    policy_net, in_keys=["observation"], out_keys=["loc", "scale"]
)

policy_module = ProbabilisticActor(
    module=policy_module,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": torch.tensor([mallet_r+0.011, mallet_r+0.011, 0, -1], dtype=torch.float32),
        "high": torch.tensor([bounds[0]/2-mallet_r-0.011, bounds[1] - mallet_r-0.011, 1, 1], dtype=torch.float32), 
    },
    default_interaction_type=tensordict.nn.InteractionType.RANDOM,
    return_log_prob=True,
).to('cuda')

#value_net = ValueDense(obs_dim)
value_net = nn.Sequential(
    nn.Linear(obs_dim, 1024),
    nn.LayerNorm(1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.LayerNorm(1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.LayerNorm(512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
)

value_net.apply(init_policy)

value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
    out_keys=["state_value"]
).to('cuda')

advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module
).to('cuda')

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=epsilon,
    entropy_bonus=True, 
    entropy_coef=entropy_coeff,
    ).to('cuda')

policy_optimizer = torch.optim.Adam(policy_module.parameters(), lr=lr_policy)
value_optimizer = torch.optim.Adam(value_module.parameters(), lr=lr_value)

if load_filepath is not None:
    checkpoint = torch.load(load_filepath, map_location='cuda')
    policy_module.load_state_dict(checkpoint['policy_state_dict'])
    value_module.load_state_dict(checkpoint['value_state_dict'])
    policy_optimizer.load_state_dict(checkpoint['optim_policy_state_dict'])
    value_optimizer.load_state_dict(checkpoint['optim_value_state_dict'])

envs = 2048 #2048
if not train:
    envs = 1

# Generate 2D low-frequency Perlin noise
mallet_init = np.array([[0.25, 0.5], [0,0]])
mallet_init[1] = bounds - mallet_init[0]

puck_init = np.array([[2*puck_r, bounds[0]/2-2*puck_r],[2*puck_r,bounds[1]-2*puck_r]])
puck_pos = []
while len(puck_pos) < envs:
    samples_needed = envs - len(puck_pos)
    samples = np.random.uniform(puck_init[:,0], puck_init[:,1], size=(samples_needed * 2, 2))
    dists = np.linalg.norm(samples - mallet_init[0], axis=1)
    valid_samples = samples[dists > mallet_r + 2*puck_r]
    puck_pos.extend(valid_samples[:samples_needed])

puck_pos = np.array(puck_pos[:envs])

obs = np.empty((2*envs, obs_dim))

shape = (2000, 1000)
res = (20, 20)  # low resolution = low frequency

noise_map= generate_perlin_noise_2d(shape, res) * sensor_error
noise_seeds = np.empty((envs, 4, 2), dtype=np.int16) # puckx pucky malletx mallety, xy coord
for i in range(4):
    noise_seeds[:,i,0] = (np.random.rand(envs) * (2000-200)).astype(np.int16)
    noise_seeds[:,i,1] = (np.random.rand(envs) * (1000-100)).astype(np.int16)

puck_noise = np.empty((envs, 2))
for i in range(2):
    noise_idx = (puck_pos*100).astype(np.int16) + noise_seeds[:,i,:]
    puck_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]

mallet_noise = np.empty((envs, 4)) #mallet, op mallet
for i in range(4):
    noise_idx = (mallet_init[int(i/2)]*100).astype(np.int16) + noise_seeds[:,2+i%2,:]
    mallet_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]

mallet_vel_noise = [-0.02, 0.02]

e_ni = [0.6, 0.9]
e_nf = 0.3
e_nr = [0, 0.25]
e_ti = 0.5
e_std = [0,0.05]

coll_vars = np.empty((envs, 14))
    
# Independent samples
coll_vars[:, [0, 7]] = np.random.uniform(e_ni[0], e_ni[1], (envs, 2))    # e_ni
coll_vars[:, [1, 8]] = np.random.uniform(e_nr[0], e_nr[1], (envs, 2))      # e_nr  
coll_vars[:, [6, 13]] = np.random.uniform(e_std[0], e_std[1], (envs, 2))    # e_std

# Dependent samples
coll_vars[:, [2, 9]] = np.random.uniform(e_nf, coll_vars[:, [0, 7]])  # e_nf
coll_vars[:, [3, 10]] = np.random.uniform(e_ti, coll_vars[:, [0, 7]]) # e_ti
coll_vars[:, [4, 11]] = np.random.uniform(coll_vars[:, [1, 8]], e_nr[1]) # e_tr
coll_vars[:, [5, 12]] = np.random.uniform(e_nf, coll_vars[:, [2, 9]]) # e_tf

#require C6^2 > 4 * C5 * C7

#[C5x, C6x, C7x, C5y, C6y, C7y]
#[5.2955 * 10^-6, 8.449*10^-3, 
#[1.8625 * 10^-6, 2.971*10^-3, ]

#a1 = 3.579*10**(-6)
ab_ranges = np.array([
        [2.86e-6, 5.35e-6],  # a1
        [4.57e-3, 8.55e-3],  # a2
        [4.25e-2, 7.15e-2],  # a3
        [-2.19e-6, -1.38e-6], # b1
        [-3.5e-3, -2.19e-3]   # b2
    ])
#b3 is 0

ab_vars = np.random.uniform(
        low=ab_ranges[:, 0].reshape(1, 1, 5),    # Shape: (1, 1, 5)
        high=ab_ranges[:, 1].reshape(1, 1, 5),   # Shape: (1, 1, 5)
        size=(envs, 2, 5)                     # Output shape
    )

ab_vars *= np.random.uniform(0.8, 1.3, size=(envs, 2, 1))

ab_obs = np.zeros((2*envs, 5))

ab_obs_scaling = np.array([
        1e4,   # a1: ~1e-6 * 1e4 = 1e-2
        1e1,   # a2: ~1e-3 * 1e1 = 1e-2 (keep as is since it's already reasonable)
        1e0,   # a3: ~1e-2 * 1e0 = 1e-2 (keep as is)
        1e4,   # b1: ~1e-6 * 1e4 = 1e-2
        1e1    # b2: ~1e-3 * 1e1 = 1e-2 (keep as is)
    ])
    
ab_obs[:envs, :] = (ab_vars[:,0,:] / pullyR) * ab_obs_scaling[:]
ab_obs[envs:, :] = (ab_vars[:,1,:] / pullyR) * ab_obs_scaling[:]

#[puck_pos, opp_mallet_pos] #guess mean low std
#[mallet_pos, mallet_vel, past mallet_pos, past_mallet_vel, past action]
#[e_n0, e_nr, e_nf, e_t0, e_tr, e_tf, var] x2 mallet and wall,
#[a1, a2, a3, b1, b2]
#20 + 8 + 4 + 14 + 5

sim.initalize(envs=envs, mallet_r=mallet_r, puck_r=puck_r, goal_w=goal_width, V_max=Vmax, pully_radius=pullyR, coll_vars=coll_vars, ab_vars=ab_vars, puck_inits=puck_pos)


obs[:envs, :] = np.concatenate([np.tile(np.concatenate([puck_pos+puck_noise,
                                    np.full((envs,2), mallet_init[1])+mallet_noise[:,2:]], axis=1), (5,)),
                                np.full((envs,2), mallet_init[0])+mallet_noise[:,:2],
                                np.random.uniform(mallet_vel_noise[0], mallet_vel_noise[1], size=(envs,2)),
                                np.full((envs,2), mallet_init[0])+mallet_noise[:,:2],
                                np.random.uniform(mallet_vel_noise[0], mallet_vel_noise[1], size=(envs,2)),
                                np.full((envs,2), mallet_init[0]) + mallet_noise[:,:2],
                                np.random.rand(envs, 1),
                                np.random.rand(envs, 1)*2 - 1,
                                coll_vars,
                                ab_obs[:envs, :],
                                np.random.random((envs,1))], axis=1)

obs[envs:, :] = np.concatenate([np.tile(np.concatenate([bounds-puck_pos+puck_noise,
                                    np.full((envs,2), mallet_init[1])+mallet_noise[:,:2]], axis=1), (5,)),
                                np.full((envs,2), mallet_init[0])+mallet_noise[:,2:],
                                np.random.uniform(mallet_vel_noise[0], mallet_vel_noise[1], size=(envs,2)),
                                np.full((envs,2), mallet_init[0])+mallet_noise[:,2:],
                                np.random.uniform(mallet_vel_noise[0], mallet_vel_noise[1], size=(envs,2)),
                                np.full((envs,2), mallet_init[0]) + mallet_noise[:,2:],
                                np.random.rand(envs, 1),
                                np.random.rand(envs, 1)*2 - 1,
                                coll_vars,
                                ab_obs[envs:, :],
                                np.random.random((envs,1))], axis=1)

past_obs = obs.copy()
obs_init = obs[:,:len(obs[0])-20].copy()

camera_buffer_size = 20

class CircularCameraBuffer():
    def __init__(self, array_init, buffer_size):
        self.array = array_init
        self.init_array = array_init.copy()
        self.buffer_size = buffer_size
        self.head = 0

    def put(self, arr):
        self.head = (self.head - 1) % self.buffer_size
        self.array[self.head, :, :] = arr

    def reset(self, env_list):
        self.array[:, env_list, :] = self.init_array[:, env_list, :]

    def get(self, indices=[]):
        new_indicies = []
        for idx in indices:
            new_indicies.append((self.head + idx) % self.buffer_size)
        return self.array[np.array(new_indicies)].transpose(1, 0, 2).reshape(2*envs, -1)
    
camera_buffer = np.empty((camera_buffer_size, 2*envs, 4))
camera_buffer[:, :envs, :2] = puck_pos+puck_noise
camera_buffer[:, envs:, :2] = bounds-puck_pos+puck_noise
camera_buffer[:, :envs, 2:] = np.full((envs,2), mallet_init[1])+mallet_noise[:,2:]
camera_buffer[:, envs:, 2:] = np.full((envs,2), mallet_init[1])+mallet_noise[:,:2]

camera_buffer = CircularCameraBuffer(camera_buffer, camera_buffer_size)

class CircularBuffer():
    def __init__(self, array_init):
        self.array = array_init
        self.buffer_size = len(array_init)
        self.head = 0

    def subtract(self, amount):
        self.array -= amount

    def put(self, value):
        self.head = (self.head - 1) % self.buffer_size
        self.array[self.head] = value

    def get(self, idx):
        return self.array[(self.head + idx) % self.buffer_size]

time_from_last_img = 0

inference_img = np.zeros((camera_buffer_size,), dtype=np.bool_)
for i in range(len(inference_img)):
    if i % 2 == 0:
        inference_img[i] = True
inference_img = CircularBuffer(inference_img)

agent_actions = np.empty((camera_buffer_size,))
for i in range(len(agent_actions)):
    agent_actions[i] = np.clip(np.random.normal(image_delay[0], image_delay[1]), image_delay[2], image_delay[3]) - i*camera_period
agent_actions = CircularBuffer(agent_actions)

mallet_time = np.empty((camera_buffer_size,))
for i in range(len(mallet_time)):
    mallet_time[i] = agent_actions.get(i) - np.clip(np.random.normal(mallet_delay[0], mallet_delay[1]), mallet_delay[2], mallet_delay[3])
mallet_time = CircularBuffer(mallet_time)

save_num = 0

#move timeline to first action time
while True:
    img_idx = None
    next_img = time_from_last_img + camera_period
    next_mallet = np.inf
    next_action = np.inf
    for i in range(camera_buffer_size):
        if not inference_img.get(i):
            continue

        sample_mallet = mallet_time.get(i)
        if sample_mallet < 1e-8:
            break
        if sample_mallet < next_mallet:
            next_mallet = sample_mallet

    for i in range(camera_buffer_size):
        if not inference_img.get(i):
            continue

        action_time = agent_actions.get(i)
        if action_time < 1e-8:
            break
        if action_time < next_action:
            next_action = action_time
            img_idx = i


    if next_img < next_mallet and next_img < next_action:
        #mallet_pos (mallet, op_mallet)
        mallet_pos, _, puck_pos, cross_left, cross_right = sim.step(next_img)
        for i in range(2):
            noise_idx = (puck_pos*100).astype(np.int16) + noise_seeds[:,i,:]
            puck_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]
        puck_pos = np.concatenate([puck_pos + puck_noise, bounds - puck_pos + puck_noise],axis=0)

        #mallet_noise = np.empty((envs, 4)) #mallet, op mallet
        for i in range(4):
            noise_idx = (mallet_pos[:,int(i/2),:]*100).astype(np.int16) + noise_seeds[:,2+i%2,:]
            mallet_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]
        op_mallet_pos = np.concatenate([mallet_pos[:,1,:] + mallet_noise[:,2:], bounds - mallet_pos[:,0,:] + mallet_noise[:,:2]], axis=0)
        camera_buffer.put(np.concatenate([puck_pos, op_mallet_pos], axis=1))

        time_from_last_img = 0
        agent_actions.subtract(next_img)
        mallet_time.subtract(next_img)
        agent_actions.put(np.clip(np.random.normal(image_delay[0], image_delay[1]), image_delay[2], image_delay[3]))
        mallet_time.put(agent_actions.get(0) - np.clip(np.random.normal(mallet_delay[0], mallet_delay[1]), mallet_delay[2], mallet_delay[3]))

        inference_img.put(np.logical_not(inference_img.get(0)))
    elif next_mallet < next_img and next_mallet < next_action:
        mallet_pos, mallet_vel, _, cross_left, cross_right = sim.step(next_mallet)

        #mallet_noise = np.empty((envs, 4)) #mallet, op mallet
        for i in range(4):
            noise_idx = (mallet_pos[:,int(i/2),:]*100).astype(np.int16) + noise_seeds[:,2+i%2,:]
            mallet_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]
        #(2*envs, 2)
        past_obs[:,20:22] = np.concatenate([mallet_pos[:,0,:] + mallet_noise[:,:2], bounds - mallet_pos[:,1,:] + mallet_noise[:,2:]], axis=0)
        vel_noise = np.random.uniform(mallet_vel_noise[0], mallet_vel_noise[1], size=(envs,2,2))
        past_obs[:,22:24] = np.concatenate([mallet_vel[:,0,:] + vel_noise[:,0,:], -mallet_vel[:,1,:] + vel_noise[:,1,:]], axis=0)
        
        time_from_last_img -= next_mallet
        agent_actions.subtract(next_mallet)
        mallet_time.subtract(next_mallet)

    elif next_action < next_img and next_action < next_mallet:
        _, _, _, cross_left, cross_right = sim.step(next_action)

        #[puck_pos, opp_mallet_pos] #guess mean low std
        #[mallet_pos, mallet_vel, past mallet_pos, past_mallet_vel, past action (x0, V)]
        #[e_n0, e_nr, e_nf, e_t0, e_tr, e_tf, var] x2 mallet and wall,

        # (2*envs, 20)
        camera_obs = camera_buffer.get(indices=[img_idx, img_idx+1, img_idx+2, img_idx+5, img_idx+11])
        past_obs[:,:20] = camera_obs

        time_from_last_img -= next_action
        agent_actions.subtract(next_action)
        mallet_time.subtract(next_action)

        break

dones = np.zeros((2*envs,), dtype=np.bool_)
rewards = np.zeros((2*envs,))

xf_mallet_noise = np.empty((envs, 4))
miss_penalty = 5.0
actions_since_cross = np.zeros((envs,), dtype=np.int32)

if train:

    buffer_size = 50_000
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(buffer_size)
    )

    #Simulation loop
    for update in range(500000):  # Training loop
        print(update)
        print("simulating...")
        for timestep in range(13):
            tensor_obs = TensorDict({"observation": torch.tensor(past_obs, dtype=torch.float32)}).to('cuda')

            policy_out = policy_module(tensor_obs)
            actions = policy_out["action"].detach().to('cpu')
            policy_out["sample_log_prob"] = torch.maximum(policy_out["sample_log_prob"], torch.tensor(-8, dtype=torch.float32))
            log_prob = policy_out["sample_log_prob"].detach().to('cpu')

            actions_np = actions.numpy()

            obs[:, 24:28] = past_obs[:, 20:24]
            obs[:, 28:32] = actions_np

            xf = np.stack([actions_np[:envs, :2], actions_np[envs:,:2]], axis=1)
            xf[:,1,:] = bounds - xf[:,1,:]

                #mallet, op mallet
            for i in range(4):
                noise_idx = (xf[:,int(i/2),:]*100).astype(np.int16) + noise_seeds[:,2+i%2,:]
                xf_mallet_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]
            xf[:,0,:] -= xf_mallet_noise[:,:2]
            xf[:,1,:] += xf_mallet_noise[:,2:]

            xf[:,0,0] = np.clip(xf[:,0,0], mallet_r+1e-4, bounds[0]/2-mallet_r-1e-4)
            xf[:,1,0] = np.clip(xf[:,1,0], bounds[0]/2+mallet_r+1e-4, bounds[0]-mallet_r-1e-4)
            xf[:,:,1] = np.clip(xf[:,:,1], mallet_r+1e-4, bounds[1]-mallet_r-1e-4)

            Vo = actions_np[:,2][:, None] * Vmax * np.stack((1+actions_np[:,3], 1-actions_np[:,3]), axis=1)
            too_low_voltage = np.logical_or(Vo[:,0] < 0.3, Vo[:,1] < 0.3)
            rewards[too_low_voltage] -= 0.5
            if too_low_voltage[0]:
                print("LOW VOLTAGE")
            Vo = np.stack([Vo[:envs,:], Vo[envs:,:]], axis=1)

            sim.take_action(xf, Vo)

            midpoint_acc = sim.get_xpp(0.004)
            rewards[:envs] -= np.sqrt(np.sum(midpoint_acc[:,0,:]**2, axis=1)) / 1000
            rewards[envs:] -= np.sqrt(np.sum(midpoint_acc[:,1,:]**2, axis=1)) / 1000

            #acc_time = sim.get_a()
            #acc_time = np.concatenate([np.max(acc_time[:,0,:],axis=1), np.max(acc_time[:,1,:],axis=1)])
            #rewards += np.clip(0.05 - acc_time, -10, 0) / 5000

            #on_edge_mask = np.logical_or(np.logical_or(xf[:,0,0] < 0.02+mallet_r, xf[:,0,0] > bounds[0]/2-mallet_r - 0.02), np.logical_or(xf[:,0,1] < mallet_r + 0.02, xf[:,0,1] > bounds[1]-mallet_r - 0.02))
            #rewards[:envs][on_edge_mask] -= 0.01

            #on_edge_mask = np.logical_or(np.logical_or(xf[:,1,0] < bounds[0]/2+mallet_r + 0.02, xf[:,1,0] > bounds[0] -mallet_r- 0.02), np.logical_or(xf[:,1,1] < mallet_r+0.02, xf[:,1,1] > bounds[1]-mallet_r - 0.02))
            #rewards[envs:][on_edge_mask] -= 0.01

            #puck_left = obs[:envs,0] < bounds[0]/2

            rewards[:envs] += np.where(np.linalg.norm(actions_np[:envs,:2] - obs[:envs,:2], axis=1) < 0.1, 1, 0)
            rewards[envs:] += np.where(np.linalg.norm(actions_np[envs:,:2] - obs[envs:,:2], axis=1) < 0.1, 1, 0)

            #print(rewards[0])

            while True:
                img_idx = None
                next_img = time_from_last_img + camera_period
                next_mallet = np.inf
                next_action = np.inf
                for i in range(camera_buffer_size):
                    if not inference_img.get(i):
                        continue

                    sample_mallet = mallet_time.get(i)
                    if sample_mallet < 1e-8:
                        break
                    if sample_mallet < next_mallet:
                        next_mallet = sample_mallet

                for i in range(camera_buffer_size):
                    if not inference_img.get(i):
                        continue

                    action_time = agent_actions.get(i)
                    if action_time < 1e-8:
                        break
                    if action_time < next_action:
                        next_action = action_time
                        img_idx = i


                if next_img < next_mallet and next_img < next_action:
                    #mallet_pos (mallet, op_mallet)
                    mallet_pos, _, puck_pos, cross_left, cross_right = sim.step(next_img)

                    env_err = sim.check_state()
                    entered_left_goal_mask, entered_right_goal_mask = sim.check_goal()
                    if len(env_err) > 0:
                        print("err")
                        print(env_err)
                        for idx in env_err:
                            dones[idx] = True
                            dones[envs+idx] = True
                            rewards[idx] -= -100.0
                            rewards[envs+idx] -= -100.0

                    dones[:envs][entered_left_goal_mask | entered_right_goal_mask] = True
                    dones[envs:][entered_left_goal_mask | entered_right_goal_mask] = True

                    rewards[:envs][entered_left_goal_mask] -= 100
                    rewards[:envs][entered_right_goal_mask] += 100
                    rewards[envs:][entered_left_goal_mask] += 100
                    rewards[envs:][entered_right_goal_mask] -= 100

                    rewards[:envs] += np.where(cross_right > 0,\
                                    cross_right / 2,\
                                    np.where(cross_right == -0.5, miss_penalty, 0))
                    rewards[envs:] += np.where(cross_left > 0,\
                                    cross_left / 2,\
                                    np.where(cross_left == -0.5, miss_penalty, 0))
                    
                    actions_since_cross[np.logical_or(cross_left != -1, cross_right != -1)] = 0

                    for i in range(2):
                        noise_idx = (puck_pos*100).astype(np.int16) + noise_seeds[:,i,:]
                        noise_idx[:,0] = np.clip(noise_idx[:,0], 0, 1999)
                        noise_idx[:,1] = np.clip(noise_idx[:,1], 0, 999)
                        puck_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]
                    puck_pos = np.concatenate([puck_pos + puck_noise, bounds - puck_pos + puck_noise],axis=0)

                    #mallet_noise = np.empty((envs, 4)) #mallet, op mallet
                    for i in range(4):
                        noise_idx = (mallet_pos[:,int(i/2),:]*100).astype(np.int16) + noise_seeds[:,2+i%2,:]
                        mallet_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]
                    op_mallet_pos = np.concatenate([mallet_pos[:,1,:] + mallet_noise[:,2:], bounds - mallet_pos[:,0,:] + mallet_noise[:,:2]], axis=0)
                    camera_buffer.put(np.concatenate([puck_pos, op_mallet_pos], axis=1))

                    time_from_last_img = 0
                    agent_actions.subtract(next_img)
                    mallet_time.subtract(next_img)
                    agent_actions.put(np.clip(np.random.normal(image_delay[0], image_delay[1]), image_delay[2], image_delay[3]))
                    mallet_time.put(agent_actions.get(0) - np.clip(np.random.normal(mallet_delay[0], mallet_delay[1]), mallet_delay[2], mallet_delay[3]))

                    inference_img.put(np.logical_not(inference_img.get(0)))
                elif next_mallet < next_img and next_mallet < next_action:
                    mallet_pos, mallet_vel, _, cross_left, cross_right = sim.step(next_mallet)

                    env_err = sim.check_state()
                    entered_left_goal_mask, entered_right_goal_mask = sim.check_goal()
                    if len(env_err) > 0:
                        print("err")
                        print(env_err)
                        for idx in env_err:
                            dones[idx] = True
                            dones[envs+idx] = True
                            rewards[idx] -= -100.0
                            rewards[envs+idx] -= -100.0
                    dones[:envs][entered_left_goal_mask | entered_right_goal_mask] = True
                    dones[envs:][entered_left_goal_mask | entered_right_goal_mask] = True

                    rewards[:envs][entered_left_goal_mask] -= 100
                    rewards[:envs][entered_right_goal_mask] += 100
                    rewards[envs:][entered_left_goal_mask] += 100
                    rewards[envs:][entered_right_goal_mask] -= 100

                    rewards[:envs] += np.where(cross_right > 0,\
                                    cross_right / 2,\
                                    np.where(cross_right == -0.5, miss_penalty, 0))
                    rewards[envs:] += np.where(cross_left > 0,\
                                    cross_left / 2,\
                                    np.where(cross_left == -0.5, miss_penalty, 0))
                    
                    actions_since_cross[np.logical_or(cross_left!=-1, cross_right!=-1)] = 0

                    #mallet_noise = np.empty((envs, 4)) #mallet, op mallet
                    for i in range(4):
                        noise_idx = (mallet_pos[:,int(i/2),:]*100).astype(np.int16) + noise_seeds[:,2+i%2,:]
                        mallet_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]
                    #(2*envs, 2)
                    obs[:,20:22] = np.concatenate([mallet_pos[:,0,:] + mallet_noise[:,:2], bounds - mallet_pos[:,1,:] + mallet_noise[:,2:]], axis=0)
                    vel_noise = np.random.uniform(mallet_vel_noise[0], mallet_vel_noise[1], size=(envs,2,2))
                    obs[:,22:24] = np.concatenate([mallet_vel[:,0,:] + vel_noise[:,0,:], -mallet_vel[:,1,:] + vel_noise[:,1,:]], axis=0)
                    
                    time_from_last_img -= next_mallet
                    agent_actions.subtract(next_mallet)
                    mallet_time.subtract(next_mallet)

                elif next_action < next_img and next_action < next_mallet:
                    _, _, puck_pos, cross_left, cross_right = sim.step(next_action)

                    env_err = sim.check_state()
                    entered_left_goal_mask, entered_right_goal_mask = sim.check_goal()
                    if len(env_err) > 0:
                        print("err")
                        print(env_err)
                        for idx in env_err:
                            dones[idx] = True
                            dones[envs+idx] = True
                            rewards[idx] -= -100.0
                            rewards[envs+idx] -= -100.0
                    dones[:envs][entered_left_goal_mask | entered_right_goal_mask] = True
                    dones[envs:][entered_left_goal_mask | entered_right_goal_mask] = True

                    rewards[:envs][entered_left_goal_mask] -= 100
                    rewards[:envs][entered_right_goal_mask] += 100
                    rewards[envs:][entered_left_goal_mask] += 100
                    rewards[envs:][entered_right_goal_mask] -= 100

                    rewards[:envs] += np.where(cross_right > 0,\
                                    cross_right / 2,\
                                    np.where(cross_right == -0.5, miss_penalty, 0))
                    rewards[envs:] += np.where(cross_left > 0,\
                                    cross_left / 2,\
                                    np.where(cross_left == -0.5, miss_penalty, 0))
                    
                    puck_left = puck_pos[:,0] < (bounds[0]/2)
                    #obs[:envs,-1][puck_left] += 1/(60)
                    #obs[:envs,-1][np.logical_not(puck_left)] = 0
                    #obs[envs:,-1][np.logical_not(puck_left)] += 1/(60)
                    #obs[envs:,-1][puck_left] = 0
                    rewards[:envs][np.logical_not(puck_left)] -= 2*(np.linalg.norm(obs[:envs,20:22] - mallet_init[0], axis=1))[np.logical_not(puck_left)]
                    rewards[envs:][puck_left] -= 2*(np.linalg.norm(obs[envs:,20:22] - mallet_init[0],axis=1))[puck_left]

                    obs[:,-1] += 1/(horizon)

                    actions_since_cross[np.logical_or(cross_left!=-1, cross_right!=-1)] = 0

                    # (2*envs, 20)
                    camera_obs = camera_buffer.get(indices=[img_idx, img_idx+1, img_idx+2, img_idx+5, img_idx+11])
                    obs[:,:20] = camera_obs

                    time_from_last_img -= next_action
                    agent_actions.subtract(next_action)
                    mallet_time.subtract(next_action)

                    break

            #timer_reset = np.logical_or(obs[:envs,-1] > 0.999, obs[envs:,-1] > 0.999)
            #dones[:envs][timer_reset] = True
            #dones[envs:][timer_reset] = True
            #rewards[obs[:,-1] > 0.999] -= 10.0
            out_of_time = np.logical_and(obs[:,-1] >= 1, np.logical_not(dones))
            dones[out_of_time] = True
            
            replay_buffer.extend(TensorDict({
                    "observation": tensor_obs["observation"].to('cpu'),
                    "action": torch.tensor(obs[:,28:32], dtype=torch.float32),
                    "reward": torch.tensor(rewards, dtype=torch.float32),
                    "done": torch.tensor(dones, dtype=torch.bool),
                    "next_observation": torch.tensor(obs, dtype=torch.float32),
                    "sample_log_prob": log_prob
                }, batch_size=[2*envs]))
            
            sim.display_state(0)

            obs[:,-1][out_of_time] = 0
            dones[out_of_time] = False

            actions_since_cross += 1
            actions_since_cross[dones[:envs]] = 0
            too_long_reset = actions_since_cross > 60*5
            actions_since_cross[too_long_reset] = 0
            dones[:envs][too_long_reset] = True
            dones[envs:][too_long_reset] = True

            num_resets = int(np.sum(dones) / 2)

            if num_resets > 0:
                coll_vars = np.empty((num_resets, 14))

                # Independent samples
                coll_vars[:, [0, 7]] = np.random.uniform(e_ni[0], e_ni[1], (num_resets, 2))    # e_ni
                coll_vars[:, [1, 8]] = np.random.uniform(e_nr[0], e_nr[1], (num_resets, 2))      # e_nr  
                coll_vars[:, [6, 13]] = np.random.uniform(e_std[0], e_std[1], (num_resets, 2))    # e_std

                # Dependent samples
                coll_vars[:, [2, 9]] = np.random.uniform(e_nf, coll_vars[:, [0, 7]])  # e_nf
                coll_vars[:, [3, 10]] = np.random.uniform(e_ti, coll_vars[:, [0, 7]]) # e_ti
                coll_vars[:, [4, 11]] = np.random.uniform(coll_vars[:, [1, 8]], e_nr[1]) # e_tr
                coll_vars[:, [5, 12]] = np.random.uniform(e_nf, coll_vars[:, [2, 9]]) # e_tf

                ab_vars = np.random.uniform(
                        low=ab_ranges[:, 0].reshape(1, 1, 5),    # Shape: (1, 1, 5)
                        high=ab_ranges[:, 1].reshape(1, 1, 5),   # Shape: (1, 1, 5)
                        size=(num_resets, 2, 5)                     # Output shape
                    )

                ab_vars *= np.random.uniform(0.8, 1.3, size=(num_resets, 2, 1))

                ab_obs = np.zeros((2*num_resets, 5))
                    
                ab_obs[:num_resets, :] = (ab_vars[:,0,:] / pullyR) * ab_obs_scaling[:]
                ab_obs[num_resets:, :] = (ab_vars[:,1,:] / pullyR) * ab_obs_scaling[:]

                obs[dones] = np.concatenate([obs_init[dones], np.tile(coll_vars, (2,1)), ab_obs, np.random.random((2*num_resets, 1))], axis=1)

                camera_buffer.reset(np.where(dones)[0])

                sim.reset_sim(np.where(dones[:envs])[0], coll_vars, ab_vars)

            past_obs = obs.copy()
            dones[:] = False
            rewards[:] = 0

        if update < 6:
            continue
                
        print("training...")
        for i in range(25):
            batch = replay_buffer.sample(batch_size).to('cuda')
            
            t_obs = batch["observation"]
            t_action = batch["action"]
            t_reward = batch["reward"]
            t_done = batch["done"]
            t_next_obs = batch["next_observation"]
            t_log_prob = batch["sample_log_prob"]

            #[puck_pos, opp_mallet_pos] #guess mean low std
            #[mallet_pos, mallet_vel, past mallet_pos, past_mallet_vel, xf, vo]
            #[e_n0, e_nr, e_nf, e_t0, e_tr, e_tf, var] x2 mallet and wall,
            #[a1, a2, a3, b1, b2]

            t_obs_flip = t_obs.clone().detach()
            t_obs_flip[:,1:22:2] = bounds[1] - t_obs_flip[:, 1:22:2]
            t_obs_flip[:, 23] *= -1
            t_obs_flip[:, 27] *= -1
            t_obs_flip[:, 25] = bounds[1] - t_obs_flip[:, 25]
            t_obs_flip[:, 29] = bounds[1] - t_obs_flip[:, 29]

            t_action_flip = t_action.clone().detach()
            t_action_flip[:,1] = bounds[1] - t_action_flip[:,1]

            t_next_obs_flip = t_next_obs.clone().detach()
            t_next_obs_flip[:,1:22:2] = bounds[1] - t_next_obs_flip[:, 1:22:2]
            t_next_obs_flip[:, 23] *= -1
            t_next_obs_flip[:, 27] *= -1
            t_next_obs_flip[:, 25] = bounds[1] - t_next_obs_flip[:, 25]
            t_next_obs_flip[:, 29] = bounds[1] - t_next_obs_flip[:, 29]

            tensordict_data = TensorDict({
                "action": t_action,
                "observation": t_obs,
                "next": TensorDict({
                    "done": t_done.clone().detach(),
                    "observation": t_next_obs,
                    "reward": t_reward.clone().detach(),
                    "terminated": t_done.clone().detach()
                }, batch_size=[batch_size]),
                "sample_log_prob": t_log_prob.clone().detach()
            }, batch_size=[batch_size]).to('cuda')

            with torch.no_grad():
                advantage_module(tensordict_data)

            loss_vals = loss_module(tensordict_data)
            #loss_value = (
            #    loss_vals["loss_objective"]
            #    + loss_vals["loss_critic"]
            #    + loss_vals["loss_entropy"]
            #)

            #if update > 30:
            policy_optimizer.zero_grad()
            loss_vals["loss_objective"].backward()
            torch.nn.utils.clip_grad_norm_(policy_module.parameters(),1)
            policy_optimizer.step()

            value_optimizer.zero_grad()
            loss_vals["loss_critic"].backward()
            torch.nn.utils.clip_grad_norm_(value_module.parameters(),1)
            value_optimizer.step()

            #optimizer.zero_grad()
            #loss_value.backward()
            #torch.nn.utils.clip_grad_norm_(loss_module.parameters(),1)
            #optimizer.step()

            tensordict_data = TensorDict({
                "action": t_action_flip,
                "observation": t_obs_flip,
                "next": TensorDict({
                    "done": t_done.clone().detach(),
                    "observation": t_next_obs_flip,
                    "reward": t_reward.clone().detach(),
                    "terminated": t_done.clone().detach()
                }, batch_size=[batch_size]),
                "sample_log_prob": t_log_prob.clone().detach()
            }, batch_size=[batch_size]).to('cuda')

            with torch.no_grad():
                advantage_module(tensordict_data)

            loss_vals = loss_module(tensordict_data)
            
            #if update > 30:
            policy_optimizer.zero_grad()
            loss_vals["loss_objective"].backward()
            torch.nn.utils.clip_grad_norm_(policy_module.parameters(),1)
            policy_optimizer.step()

            value_optimizer.zero_grad()
            loss_vals["loss_critic"].backward()
            torch.nn.utils.clip_grad_norm_(value_module.parameters(),1)
            value_optimizer.step()

        if update % 25 == 0:
            torch.save({
                'policy_state_dict': policy_module.state_dict(),
                'value_state_dict': value_module.state_dict(),
                'optim_policy_state_dict': policy_optimizer.state_dict(),
                'optim_value_state_dict': value_optimizer.state_dict(),
            }, save_filepath)

reward_sum = np.zeros((2*envs))
for timestep in range(100000):
    tensor_obs = TensorDict({"observation": torch.tensor(past_obs, dtype=torch.float32)}).to('cuda')

    policy_out = policy_module(tensor_obs)
    value_out = value_module(tensor_obs)
    print("---")
    print(value_out["state_value"].detach().to('cpu').numpy())
    actions = policy_out["action"].detach().to('cpu')
    policy_out["sample_log_prob"] = torch.maximum(policy_out["sample_log_prob"], torch.tensor(-8, dtype=torch.float32))
    log_prob = policy_out["sample_log_prob"].detach().to('cpu')

    actions_np = actions.numpy()

    obs[:, 24:28] = past_obs[:, 20:24]
    obs[:, 28:32] = actions_np

    xf = np.stack([actions_np[:envs, :2], actions_np[envs:,:2]], axis=1)
    xf[:,1,:] = bounds - xf[:,1,:]

        #mallet, op mallet
    for i in range(4):
        noise_idx = (xf[:,int(i/2),:]*100).astype(np.int16) + noise_seeds[:,2+i%2,:]
        xf_mallet_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]
    xf[:,0,:] -= xf_mallet_noise[:,:2]
    xf[:,1,:] += xf_mallet_noise[:,2:]

    xf[:,0,0] = np.clip(xf[:,0,0], mallet_r+1e-4, bounds[0]/2-mallet_r-1e-4)
    xf[:,1,0] = np.clip(xf[:,1,0], bounds[0]/2+mallet_r+1e-4, bounds[0]-mallet_r-1e-4)
    xf[:,:,1] = np.clip(xf[:,:,1], mallet_r+1e-4, bounds[1]-mallet_r-1e-4)

    Vo = actions_np[:,2][:, None] * Vmax * np.stack((1+actions_np[:,3], 1-actions_np[:,3]), axis=1)
    too_low_voltage = np.logical_or(Vo[:,0] < 0.3, Vo[:,1] < 0.3)
    rewards[too_low_voltage] -= 0.5
    if too_low_voltage[0]:
        print("LOW VOLTAGE")
    Vo = np.stack([Vo[:envs,:], Vo[envs:,:]], axis=1)

    sim.take_action(xf, Vo)

    midpoint_acc = sim.get_xpp(0.004)
    rewards[:envs] -= np.sqrt(np.sum(midpoint_acc[:,0,:]**2, axis=1)) / 1000
    rewards[envs:] -= np.sqrt(np.sum(midpoint_acc[:,1,:]**2, axis=1)) / 1000

    #acc_time = sim.get_a()
    #acc_time = np.concatenate([np.max(acc_time[:,0,:],axis=1), np.max(acc_time[:,1,:],axis=1)])
    #rewards += np.clip(0.05 - acc_time, -10, 0) / 5000

    #on_edge_mask = np.logical_or(np.logical_or(xf[:,0,0] < 0.02+mallet_r, xf[:,0,0] > bounds[0]/2-mallet_r - 0.02), np.logical_or(xf[:,0,1] < mallet_r + 0.02, xf[:,0,1] > bounds[1]-mallet_r - 0.02))
    #rewards[:envs][on_edge_mask] -= 0.01

    #on_edge_mask = np.logical_or(np.logical_or(xf[:,1,0] < bounds[0]/2+mallet_r + 0.02, xf[:,1,0] > bounds[0] -mallet_r- 0.02), np.logical_or(xf[:,1,1] < mallet_r+0.02, xf[:,1,1] > bounds[1]-mallet_r - 0.02))
    #rewards[envs:][on_edge_mask] -= 0.01

    #puck_left = obs[:envs,0] < bounds[0]/2

    rewards[:envs] += np.where(np.linalg.norm(actions_np[:envs,:2] - obs[:envs,:2], axis=1) < 0.1, 1, 0)
    rewards[envs:] += np.where(np.linalg.norm(actions_np[envs:,:2] - obs[envs:,:2], axis=1) < 0.1, 1, 0)

    #print(rewards[0])

    while True:
        img_idx = None
        next_img = time_from_last_img + camera_period
        next_mallet = np.inf
        next_action = np.inf
        for i in range(camera_buffer_size):
            if not inference_img.get(i):
                continue

            sample_mallet = mallet_time.get(i)
            if sample_mallet < 1e-8:
                break
            if sample_mallet < next_mallet:
                next_mallet = sample_mallet

        for i in range(camera_buffer_size):
            if not inference_img.get(i):
                continue

            action_time = agent_actions.get(i)
            if action_time < 1e-8:
                break
            if action_time < next_action:
                next_action = action_time
                img_idx = i


        if next_img < next_mallet and next_img < next_action:
            #mallet_pos (mallet, op_mallet)
            mallet_pos, _, puck_pos, cross_left, cross_right = sim.step(next_img)

            env_err = sim.check_state()
            entered_left_goal_mask, entered_right_goal_mask = sim.check_goal()
            if len(env_err) > 0:
                print("err")
                print(env_err)
                for idx in env_err:
                    dones[idx] = True
                    dones[envs+idx] = True
                    rewards[idx] -= -100.0
                    rewards[envs+idx] -= -100.0

            dones[:envs][entered_left_goal_mask | entered_right_goal_mask] = True
            dones[envs:][entered_left_goal_mask | entered_right_goal_mask] = True

            rewards[:envs][entered_left_goal_mask] -= 100
            rewards[:envs][entered_right_goal_mask] += 100
            rewards[envs:][entered_left_goal_mask] += 100
            rewards[envs:][entered_right_goal_mask] -= 100

            rewards[:envs] += np.where(cross_right > 0,\
                            cross_right / 2,\
                            np.where(cross_right == -0.5, miss_penalty, 0))
            rewards[envs:] += np.where(cross_left > 0,\
                            cross_left / 2,\
                            np.where(cross_left == -0.5, miss_penalty, 0))
            
            actions_since_cross[np.logical_or(cross_left != -1, cross_right != -1)] = 0

            for i in range(2):
                noise_idx = (puck_pos*100).astype(np.int16) + noise_seeds[:,i,:]
                noise_idx[:,0] = np.clip(noise_idx[:,0], 0, 1999)
                noise_idx[:,1] = np.clip(noise_idx[:,1], 0, 999)
                puck_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]
            puck_pos = np.concatenate([puck_pos + puck_noise, bounds - puck_pos + puck_noise],axis=0)

            #mallet_noise = np.empty((envs, 4)) #mallet, op mallet
            for i in range(4):
                noise_idx = (mallet_pos[:,int(i/2),:]*100).astype(np.int16) + noise_seeds[:,2+i%2,:]
                mallet_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]
            op_mallet_pos = np.concatenate([mallet_pos[:,1,:] + mallet_noise[:,2:], bounds - mallet_pos[:,0,:] + mallet_noise[:,:2]], axis=0)
            camera_buffer.put(np.concatenate([puck_pos, op_mallet_pos], axis=1))

            time_from_last_img = 0
            agent_actions.subtract(next_img)
            mallet_time.subtract(next_img)
            agent_actions.put(np.clip(np.random.normal(image_delay[0], image_delay[1]), image_delay[2], image_delay[3]))
            mallet_time.put(agent_actions.get(0) - np.clip(np.random.normal(mallet_delay[0], mallet_delay[1]), mallet_delay[2], mallet_delay[3]))

            inference_img.put(np.logical_not(inference_img.get(0)))
        elif next_mallet < next_img and next_mallet < next_action:
            mallet_pos, mallet_vel, _, cross_left, cross_right = sim.step(next_mallet)

            env_err = sim.check_state()
            entered_left_goal_mask, entered_right_goal_mask = sim.check_goal()
            if len(env_err) > 0:
                print("err")
                print(env_err)
                for idx in env_err:
                    dones[idx] = True
                    dones[envs+idx] = True
                    rewards[idx] -= -100.0
                    rewards[envs+idx] -= -100.0
            dones[:envs][entered_left_goal_mask | entered_right_goal_mask] = True
            dones[envs:][entered_left_goal_mask | entered_right_goal_mask] = True

            rewards[:envs][entered_left_goal_mask] -= 100
            rewards[:envs][entered_right_goal_mask] += 100
            rewards[envs:][entered_left_goal_mask] += 100
            rewards[envs:][entered_right_goal_mask] -= 100

            rewards[:envs] += np.where(cross_right > 0,\
                            cross_right / 2,\
                            np.where(cross_right == -0.5, miss_penalty, 0))
            rewards[envs:] += np.where(cross_left > 0,\
                            cross_left / 2,\
                            np.where(cross_left == -0.5, miss_penalty, 0))
            
            actions_since_cross[np.logical_or(cross_left!=-1, cross_right!=-1)] = 0

            #mallet_noise = np.empty((envs, 4)) #mallet, op mallet
            for i in range(4):
                noise_idx = (mallet_pos[:,int(i/2),:]*100).astype(np.int16) + noise_seeds[:,2+i%2,:]
                mallet_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]
            #(2*envs, 2)
            obs[:,20:22] = np.concatenate([mallet_pos[:,0,:] + mallet_noise[:,:2], bounds - mallet_pos[:,1,:] + mallet_noise[:,2:]], axis=0)
            vel_noise = np.random.uniform(mallet_vel_noise[0], mallet_vel_noise[1], size=(envs,2,2))
            obs[:,22:24] = np.concatenate([mallet_vel[:,0,:] + vel_noise[:,0,:], -mallet_vel[:,1,:] + vel_noise[:,1,:]], axis=0)
            
            time_from_last_img -= next_mallet
            agent_actions.subtract(next_mallet)
            mallet_time.subtract(next_mallet)

        elif next_action < next_img and next_action < next_mallet:
            _, _, puck_pos, cross_left, cross_right = sim.step(next_action)

            env_err = sim.check_state()
            entered_left_goal_mask, entered_right_goal_mask = sim.check_goal()
            if len(env_err) > 0:
                print("err")
                print(env_err)
                for idx in env_err:
                    dones[idx] = True
                    dones[envs+idx] = True
                    rewards[idx] -= -100.0
                    rewards[envs+idx] -= -100.0
            dones[:envs][entered_left_goal_mask | entered_right_goal_mask] = True
            dones[envs:][entered_left_goal_mask | entered_right_goal_mask] = True

            rewards[:envs][entered_left_goal_mask] -= 100
            rewards[:envs][entered_right_goal_mask] += 100
            rewards[envs:][entered_left_goal_mask] += 100
            rewards[envs:][entered_right_goal_mask] -= 100

            rewards[:envs] += np.where(cross_right > 0,\
                            cross_right / 2,\
                            np.where(cross_right == -0.5, miss_penalty, 0))
            rewards[envs:] += np.where(cross_left > 0,\
                            cross_left / 2,\
                            np.where(cross_left == -0.5, miss_penalty, 0))
            
            puck_left = puck_pos[:,0] < (bounds[0]/2)
            #obs[:envs,-1][puck_left] += 1/(60)
            #obs[:envs,-1][np.logical_not(puck_left)] = 0
            #obs[envs:,-1][np.logical_not(puck_left)] += 1/(60)
            #obs[envs:,-1][puck_left] = 0
            rewards[:envs][np.logical_not(puck_left)] -= 2*(np.linalg.norm(obs[:envs,20:22] - mallet_init[0], axis=1))[np.logical_not(puck_left)]
            rewards[envs:][puck_left] -= 2*(np.linalg.norm(obs[envs:,20:22] - mallet_init[0],axis=1))[puck_left]

            obs[:,-1] += 1/(horizon)

            actions_since_cross[np.logical_or(cross_left!=-1, cross_right!=-1)] = 0

            # (2*envs, 20)
            camera_obs = camera_buffer.get(indices=[img_idx, img_idx+1, img_idx+2, img_idx+5, img_idx+11])
            obs[:,:20] = camera_obs

            time_from_last_img -= next_action
            agent_actions.subtract(next_action)
            mallet_time.subtract(next_action)

            break

    #timer_reset = np.logical_or(obs[:envs,-1] > 0.999, obs[envs:,-1] > 0.999)
    #dones[:envs][timer_reset] = True
    #dones[envs:][timer_reset] = True
    #rewards[obs[:,-1] > 0.999] -= 10.0
    out_of_time = np.logical_and(obs[:,-1] >= 1, np.logical_not(dones))
    dones[out_of_time] = True

    reward_sum += rewards
    print(reward_sum)

 
    #replay_buffer.extend(TensorDict({
    #        "observation": tensor_obs["observation"].to('cpu'),
    #        "action": torch.tensor(obs[:,28:32], dtype=torch.float32),
    #        "reward": torch.tensor(rewards, dtype=torch.float32),
    #        "done": torch.tensor(dones, dtype=torch.bool),
    #        "next_observation": torch.tensor(obs, dtype=torch.float32),
    #        "sample_log_prob": log_prob
    #    }, batch_size=[2*envs]))
    print("===")
    print(tensor_obs["observation"].to('cpu')[:,-1])
    print(rewards)
    print(dones)
    print(obs[:,-1])
    
    sim.display_state(0)

    obs[:,-1][out_of_time] = 0
    dones[out_of_time] = False
    reward_sum[out_of_time] = 0

    actions_since_cross += 1
    actions_since_cross[dones[:envs]] = 0
    too_long_reset = actions_since_cross > 60*5
    actions_since_cross[too_long_reset] = 0
    dones[:envs][too_long_reset] = True
    dones[envs:][too_long_reset] = True
    reward_sum[dones] = 0
    

    num_resets = int(np.sum(dones) / 2)

    if num_resets > 0:
        coll_vars = np.empty((num_resets, 14))

        # Independent samples
        coll_vars[:, [0, 7]] = np.random.uniform(e_ni[0], e_ni[1], (num_resets, 2))    # e_ni
        coll_vars[:, [1, 8]] = np.random.uniform(e_nr[0], e_nr[1], (num_resets, 2))      # e_nr  
        coll_vars[:, [6, 13]] = np.random.uniform(e_std[0], e_std[1], (num_resets, 2))    # e_std

        # Dependent samples
        coll_vars[:, [2, 9]] = np.random.uniform(e_nf, coll_vars[:, [0, 7]])  # e_nf
        coll_vars[:, [3, 10]] = np.random.uniform(e_ti, coll_vars[:, [0, 7]]) # e_ti
        coll_vars[:, [4, 11]] = np.random.uniform(coll_vars[:, [1, 8]], e_nr[1]) # e_tr
        coll_vars[:, [5, 12]] = np.random.uniform(e_nf, coll_vars[:, [2, 9]]) # e_tf

        ab_vars = np.random.uniform(
                low=ab_ranges[:, 0].reshape(1, 1, 5),    # Shape: (1, 1, 5)
                high=ab_ranges[:, 1].reshape(1, 1, 5),   # Shape: (1, 1, 5)
                size=(num_resets, 2, 5)                     # Output shape
            )

        ab_vars *= np.random.uniform(0.8, 1.3, size=(num_resets, 2, 1))

        ab_obs = np.zeros((2*num_resets, 5))
            
        ab_obs[:num_resets, :] = (ab_vars[:,0,:] / pullyR) * ab_obs_scaling[:]
        ab_obs[num_resets:, :] = (ab_vars[:,1,:] / pullyR) * ab_obs_scaling[:]

        obs[dones] = np.concatenate([obs_init[dones], np.tile(coll_vars, (2,1)), ab_obs, np.random.random((2*num_resets, 1))], axis=1)

        camera_buffer.reset(np.where(dones)[0])

        sim.reset_sim(np.where(dones[:envs])[0], coll_vars, ab_vars)

    past_obs = obs.copy()
    dones[:] = False
    rewards[:] = 0
