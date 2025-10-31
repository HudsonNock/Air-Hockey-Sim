import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import Air_hockey_sim_vectorized as sim
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, MLP, QValueActor, EGreedyWrapper
from torchrl.objectives import ClipPPOLoss, SACLoss, DQNLoss, ValueEstimators
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
import random
import torch.nn.functional as F

load_filepath = "checkpoints/model_184.pth"
second_actor_filepath = "yes"
save_filepath = "checkpoints/model_185.pth" #Best is 12, 15, 22, 44, 76, 103, 121, 134, 142, 160, 165, 178
train = True
# Simulation parameters
obs_dim = 38 #[puck_pos, opp_mallet_pos] #guess mean low stdl
            #[mallet_pos, mallet_vel, past mallet_pos, past malet_vel, past action]
            #[a1, a2, a3, b1, b2, b3]
            # 20 + 8 + 4 + 6 = 38

mallet_r = 0.1011 / 2
puck_r = 0.0629 / 2
pullyR = 0.3573695

#col: n_f + (1-n_f/n_0) * 2/(1+e^(n_r x^2)) n_0

action_dim = 4
Vmax = 24*0.8

puck_std = np.array([[0.000054,0.000007], [0.000032,0.000017]]) # std in x y, a*x + b
puck_beam_std = np.array([[0.001618,0.00027], [0.000775,0.000367]]) #std in x y, a*x+b
percent_miss = np.array([0.05691475, 0.02742944]) #a*x+b
puck_perlin_std = 0.00075

camera_pos = [0.3, 1.595] #x,z
beam_width = 0.05085
beam_thickness = 0.0127
beam_height = 0.04795 #from floor roof of beam

mallet_std = 0.001
mallet_vel_std = 0.02

V_std_x = 0.3
V_std_y = 0.3

# (7.629e-06, 6.617e-03, 7e-02, -7.445e-06, -2.661e-03, 5.277e-03

ab_ranges = np.array([
        [6.35e-6, 8.5e-6],  # a1
        [5.3e-3, 8.55e-3],  # a2
        [5.25e-2, 8.15e-2],  # a3
        [-8.19e-6, -6.01e-6], # b1
        [-3.5e-3, -2.19e-3],   # b2
	    [3.5e-3, 6.5e-3] #b3
    ])

speed_var = [0.7,1.2]

#mean, std, min, max
image_delay = [15.166/1000, 0.3/1000, 14/1000, 16.5/1000]
mallet_delay = [7.17/1000, 0.3/1000, 6.2/1000, 8.0/1000]
camera_period = 1/120.0

frames = [0, 1, 2, 5, 11]

height = 1.993
width = 0.992
bounds = np.array([height, width])
goal_width = 0.254

entropy_coeff = 0.01
epsilon = 0.05
gamma = 0.997
lr_policy = 5e-6 #1e-5
lr_value = 5e-6
#lr = 1e-4
batch_sizee = 1024
num_epochs = 1
lmbda = 0.7
sim_steps = 1000
lr_defender = 5e-5
q_epsilon = 0.03
#target_kl = 0.02

#------------------

beam_coeffs = np.array([[1+(beam_height-beam_thickness)/(camera_pos[1]-beam_height+beam_thickness), (beam_height-beam_thickness)*(camera_pos[0]-beam_width/2)/(camera_pos[1]-beam_height+beam_thickness) - beam_width/2], [1+(beam_height)/(camera_pos[1]-beam_height), beam_height*(camera_pos[0]+beam_width/2)/(camera_pos[1]-beam_height) + beam_width/2]])

class ScaledNormalParamExtractor(NormalParamExtractor):
    def __init__(self, scale_factor=0.8):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        loc, scale = super().forward(x)
        #scale *= self.scale_factor
        scale = 2*self.scale_factor/ (1+torch.exp(-scale)) - self.scale_factor + 0.001
        return loc, scale
    
scale_factor = 0.8

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

action_center = torch.tensor([0, 0, 2, 0], dtype=torch.float32)

def init_policy(m):
    if isinstance(m, nn.Linear):
        if m.out_features == action_dim * 2:
            # Initialize weights to 0 so output = bias
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            m.weight.data *= 0.3
            # Bias: loc = center, log_std = 0
            bias = torch.cat([action_center, torch.ones(action_dim)*0.1], dim=0)
            with torch.no_grad():
                m.bias.copy_(bias)
        elif m.out_features == 1:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            m.weight.data *= 0.001
            with torch.no_grad():
                m.bias.copy_(torch.zeros((1,)))
        else:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

policy_net.apply(init_policy)

policy_module = TensorDictModule(
    policy_net, in_keys=["observation"], out_keys=["loc", "scale"]
)


low = torch.tensor([mallet_r+0.01, mallet_r+0.01, 0, -1], dtype=torch.float32).to('cuda')
high = torch.tensor([bounds[0]/2-mallet_r-0.01, bounds[1] - mallet_r-0.01, 1, 1], dtype=torch.float32).to('cuda')

if train:
    policy_module = ProbabilisticActor(
        module=policy_module,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": torch.tensor([mallet_r+0.01, mallet_r+0.01, 0, -1], dtype=torch.float32),
            "high": torch.tensor([bounds[0]/2-mallet_r-0.01, bounds[1] - mallet_r-0.01, 1, 1], dtype=torch.float32), 
        },
        default_interaction_type=tensordict.nn.InteractionType.RANDOM,
        return_log_prob=True,
    ).to('cuda')
else:
    policy_module = ProbabilisticActor(
        module=policy_module,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": torch.tensor([mallet_r+0.01, mallet_r+0.01, 0, -1], dtype=torch.float32),
            "high": torch.tensor([bounds[0]/2-mallet_r-0.01, bounds[1] - mallet_r-0.01, 1, 1], dtype=torch.float32), 
        },
        return_log_prob=True,
        #default_interaction_type=tensordict.nn.InteractionType.RANDOM,
    ).to('cuda')

            #"low": torch.tensor([mallet_r+0.011, mallet_r+0.011, 0, -1], dtype=torch.float32),
            #"high": torch.tensor([bounds[0]/2-mallet_r-0.011, bounds[1] - mallet_r-0.011, 1, 1],

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

q_net = MLP(
    in_features=obs_dim,
    out_features=3,
    depth=3,
    num_cells=128
)
q_module = QValueActor(q_net, in_keys=["observation"],action_space="categorical").to('cuda')
q_loss_module = DQNLoss(q_module, action_space="categorical").to('cuda')
q_loss_module.make_value_estimator(
    ValueEstimators.TD0,
    gamma=0.9)

gae_module = GAE(
        gamma=gamma,
        lmbda=lmbda,
        value_network=value_module,
        average_gae=False,  # Normalize advantages
        differentiable=False
    ).to('cuda')

#policy_optimizer = torch.optim.Adam(policy_module.parameters(), lr=lr_policy)
#value_optimizer = torch.optim.Adam(value_module.parameters(), lr=lr_value)
#value_loss_fn = nn.MSELoss()
loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=epsilon,
        entropy_bonus=True,
        entropy_coef=entropy_coeff,
        normalize_advantage=True,
        critic_coef=0.1
    ).to('cuda')

optimizer_policy = torch.optim.Adam(policy_module.parameters(), lr=lr_policy)
optimizer_value = torch.optim.Adam(value_module.parameters(), lr=lr_value)
optimizer_defender = torch.optim.Adam(q_module.parameters(), lr=lr_defender)

if load_filepath is not None:
    checkpoint = torch.load(load_filepath, map_location='cuda')
    policy_module.load_state_dict(checkpoint['policy_state_dict'])
    value_module.load_state_dict(checkpoint['value_state_dict'])
    #q_module.load_state_dict(checkpoint['q_state_dict'])
    #optimizer_policy.load_state_dict(checkpoint['policy_optim_state_dict'])
    #optimizer_value.load_state_dict(checkpoint['value_optim_state_dict'])
    #optimizer_defender.load_state_dict(checkpoint['q_optim_state_dict'])


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

noise_map= generate_perlin_noise_2d(shape, res)
noise_map = noise_map - np.mean(noise_map)
noise_map_std = np.std(noise_map)
noise_map = noise_map * (puck_perlin_std / noise_map_std)
noise_seeds = np.empty((envs, 4, 2), dtype=np.int16) # puckx pucky, malletx, mallety xy coord
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

#require C6^2 > 4 * C5 * C7

#[C5x, C6x, C7x, C5y, C6y, C7y]
#[5.2955 * 10^-6, 8.449*10^-3, 
#[1.8625 * 10^-6, 2.971*10^-3, ]

if not train:
    np.random.seed(35)

ab_vars = np.random.uniform(
        low=ab_ranges[:, 0].reshape(1, 1, 6),    # Shape: (1, 1, 6)
        high=ab_ranges[:, 1].reshape(1, 1, 6),   # Shape: (1, 1, 6)
        size=(envs, 2, 6)                     # Output shape
    )

ab_vars *= np.random.uniform(speed_var[0], speed_var[1], size=(envs, 2, 1))

if not train:
    ab_vars[0,0,:] = np.array([7.474e-06, 6.721e-03, 6.658e-02, -1.607e-06, -2.731e-03, 3.610e-03])
    #ab_vars[0,0,:] = np.array([7.629e-06, 6.617e-03, 7e-02, -7.445e-06, -2.661e-03, 5.277e-03])

ab_obs = np.zeros((2*envs, 6))

ab_obs_scaling = np.array([
        1e4,   # a1: ~1e-6 * 1e4 = 1e-2
        1e1,   # a2: ~1e-3 * 1e1 = 1e-2 (keep as is since it's already reasonable)
        1e0,   # a3: ~1e-2 * 1e0 = 1e-2 (keep as is)
        1e4,   # b1: ~1e-6 * 1e4 = 1e-2
        1e1,   # b2: ~1e-3 * 1e1 = 1e-2 (keep as is)
        1e1    # b3: ~1e-3 * 1e1 = 1e-2
    ])
    
ab_obs[:envs, :] = (ab_vars[:,0,:] / pullyR) * ab_obs_scaling[:]
ab_obs[envs:, :] = (ab_vars[:,1,:] / pullyR) * ab_obs_scaling[:]

#[puck_pos, opp_mallet_pos] #guess mean low std
#[mallet_pos, mallet_vel, past mallet_pos, past_mallet_vel, past action]
#[e_n0, e_nr, e_nf, e_t0, e_tr, e_tf, var] x2 mallet and wall,
#[a1, a2, a3, b1, b2]
#20 + 8 + 4 + 14 + 5

sim.initalize(envs=envs, mallet_r=mallet_r, puck_r=puck_r, goal_w=goal_width, V_max=Vmax, pully_radius=pullyR, ab_vars=ab_vars, puck_inits=puck_pos)

def apply_symmetry(t_obs):
    t_obs_flip = t_obs.copy()
    t_obs_flip[:, 1:22:2] = bounds[1] - t_obs_flip[:, 1:22:2]
    t_obs_flip[:, 23] *= -1
    t_obs_flip[:, 27] *= -1
    t_obs_flip[:, 25] = bounds[1] - t_obs_flip[:, 25]
    t_obs_flip[:, 29] = bounds[1] - t_obs_flip[:, 29]

    return t_obs_flip

def apply_symmetry_tensor(t_obs):
    t_obs_flip = t_obs.clone().detach()
    t_obs_flip[:, 1:22:2] = bounds[1] - t_obs_flip[:, 1:22:2]
    t_obs_flip[:, 23] *= -1
    t_obs_flip[:, 27] *= -1
    t_obs_flip[:, 25] = bounds[1] - t_obs_flip[:, 25]
    t_obs_flip[:, 29] = bounds[1] - t_obs_flip[:, 29]

    return t_obs_flip

camera_buffer_size = 30

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

beam_interference = np.empty((envs), dtype=np.bool_)
large_beam_interference = np.empty((envs), dtype=np.bool_)
beam_interference = np.logical_and(puck_pos[:,0]+puck_r > beam_coeffs[0,0] * np.full((envs,2), mallet_init[0])[:,0] + beam_coeffs[0,1], puck_pos[:,0]-puck_r < beam_coeffs[1,0] * np.full((envs,2), mallet_init[0])[:,0] + beam_coeffs[1,1])

camera_buffer[:, :envs, :2] = np.tile(puck_pos+puck_noise,(camera_buffer_size,1,1)) +\
                                     np.stack((np.random.normal(0,\
                                                np.where(beam_interference, puck_beam_std[0,0] * puck_pos[:,0] + puck_beam_std[0,1], puck_std[0,0] * puck_pos[:,0] + puck_std[0,1]),\
                                                 (camera_buffer_size,envs)),\
                                             np.random.normal(0,\
                                                np.where(beam_interference, puck_beam_std[1,0] * puck_pos[:,0] + puck_beam_std[1,1], puck_std[1,0] * puck_pos[:,0] + puck_std[1,1]),\
                                                 (camera_buffer_size,envs))),\
                                             axis=-1)

camera_buffer[:, envs:, :2] = np.tile(bounds-puck_pos, (camera_buffer_size,1,1))

camera_buffer[:, :envs, 2:] = np.tile(np.full((envs,2), mallet_init[1])+mallet_noise[:,2:], (camera_buffer_size,1,1)) + np.stack((np.random.normal(0, puck_std[0,0] * mallet_init[1,0] + puck_std[0,1], (camera_buffer_size,envs)), np.random.normal(0, puck_std[1,0] * mallet_init[1,0] + puck_std[1,1], (camera_buffer_size,envs))), axis=-1)
camera_buffer[:, envs:, 2:] = np.tile(np.full((envs,2), mallet_init[1]), (camera_buffer_size,1,1))

camera_buffer = CircularCameraBuffer(camera_buffer, camera_buffer_size)

obs[:envs, :] = np.concatenate([np.zeros((envs,20)),
                                np.full((envs,2), mallet_init[0])+np.random.normal(0,mallet_std,(envs,2)),
                                np.random.normal(0, mallet_vel_std, size=(envs,2)),
                                np.full((envs,2), mallet_init[0])+np.random.normal(0,mallet_std,(envs,2)),
                                np.random.normal(0, mallet_vel_std, size=(envs,2)),
                                np.full((envs,2), mallet_init[0]),
                                np.random.rand(envs, 1),
                                np.random.rand(envs, 1)*2 - 1,
                                ab_obs[:envs, :]], axis=1)

#puck_std = [[0,0.0005], [0,0.0005]] # std in x y, a*x + b

obs[envs:, :] = np.concatenate([np.zeros((envs,20)),
                                np.full((envs,2), mallet_init[0]),
                                np.zeros((envs,2)),
                                np.full((envs,2), mallet_init[0]),
                                np.zeros((envs,2)),
                                np.full((envs,2), mallet_init[0]),
                                np.random.rand(envs, 1),
                                np.random.rand(envs, 1)*2 - 1,
                                ab_obs[envs:, :]], axis=1)

img_idx = 4
camera_obs = camera_buffer.get(indices=[img_idx, img_idx+1, img_idx+2, img_idx+5, img_idx+11])
obs[:,:20] = camera_obs

past_obs = obs.copy()
obs_init = obs[:,:32].copy()

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
        mallet_pos, _, puck_pos, _, cross_left, cross_right = sim.step(next_img)
        for i in range(2):
            noise_idx = (puck_pos*100).astype(np.int16) + noise_seeds[:,i,:]
            noise_idx[:,0] = np.clip(noise_idx[:,0], 0, 1999)
            noise_idx[:,1] = np.clip(noise_idx[:,1], 0, 999)
            puck_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]

        beam_interference[:] = np.logical_and(puck_pos[:,0]+puck_r > beam_coeffs[0,0] * mallet_pos[:,0,0] + beam_coeffs[0,1], puck_pos[:,0]-puck_r < beam_coeffs[1,0] * mallet_pos[:,0,0] + beam_coeffs[1,1])

        puck_wgn_noise = np.stack((np.random.normal(0, np.where(beam_interference, puck_beam_std[0,0] * puck_pos[:,0] + puck_beam_std[0,1], puck_std[0,0] * puck_pos[:,0] + puck_std[0,1]), (envs,)), np.random.normal(0, np.where(beam_interference, puck_beam_std[1,0] * puck_pos[:,0] + puck_beam_std[1,1], puck_std[1,0] * puck_pos[:,0] + puck_std[1,1]), (envs,))), axis=-1)

        large_beam_interference[:] = np.logical_and(puck_pos[:,0] > beam_coeffs[0,0] * mallet_pos[:,0,0] + beam_coeffs[0,1], puck_pos[:,0] < beam_coeffs[1,0] * mallet_pos[:,0,0] + beam_coeffs[1,1])

        if large_beam_interference.any():
            past_puck_data = camera_buffer.get([0])[:envs,:2] # player, x/y (envs,2)
            miss_mask = np.random.random((envs)) < percent_miss[0] * puck_pos[:,0] + percent_miss[1] #a*x+b

            puck_pos = np.concatenate([puck_pos + puck_noise + puck_wgn_noise, bounds - puck_pos],axis=0)
            puck_pos[:envs][np.logical_and(large_beam_interference, miss_mask)] = past_puck_data[np.logical_and(large_beam_interference, miss_mask)]
        else:
            puck_pos = np.concatenate([puck_pos + puck_noise + puck_wgn_noise, bounds - puck_pos],axis=0)

        #mallet_noise = np.empty((envs, 4)) #mallet, op mallet
        for i in range(2,4):
            noise_idx = (mallet_pos[:,int(i/2),:]*100).astype(np.int16) + noise_seeds[:,2+i%2,:]
            noise_idx[:,0] = np.clip(noise_idx[:,0], 0, 1999)
            noise_idx[:,1] = np.clip(noise_idx[:,1], 0, 999)
            mallet_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]
        mallet_wgn_noise = np.stack((np.random.normal(0, puck_std[0,0] * mallet_pos[:,1,0] + puck_std[0,1], (envs,)), np.random.normal(0, puck_std[1,0] * mallet_pos[:,1,0] + puck_std[1,1], (envs,))), axis=-1)

        op_mallet_pos = np.concatenate([mallet_pos[:,1,:] + mallet_noise[:,2:] + mallet_wgn_noise, bounds - mallet_pos[:,0,:]], axis=0)
        camera_buffer.put(np.concatenate([puck_pos, op_mallet_pos], axis=1))

        time_from_last_img = 0
        agent_actions.subtract(next_img)
        mallet_time.subtract(next_img)
        agent_actions.put(np.clip(np.random.normal(image_delay[0], image_delay[1]), image_delay[2], image_delay[3]))
        mallet_time.put(agent_actions.get(0) - np.clip(np.random.normal(mallet_delay[0], mallet_delay[1]), mallet_delay[2], mallet_delay[3]))

        inference_img.put(np.logical_not(inference_img.get(0)))
    elif next_mallet < next_img and next_mallet < next_action:
        mallet_pos, mallet_vel, _, _, cross_left, cross_right = sim.step(next_mallet)

        #(2*envs, 2)
        past_obs[:,20:22] = np.concatenate([mallet_pos[:,0,:] + np.random.normal(0, mallet_std, (envs,2)), bounds - mallet_pos[:,1,:] + np.random.normal(0, mallet_std, (envs,2))], axis=0)
        vel_noise = np.random.normal(0, mallet_vel_std, size=(envs,2,2))
        past_obs[:,22:24] = np.concatenate([mallet_vel[:,0,:] + vel_noise[:,0,:], -mallet_vel[:,1,:] + vel_noise[:,1,:]], axis=0)
        
        time_from_last_img -= next_mallet
        agent_actions.subtract(next_mallet)
        mallet_time.subtract(next_mallet)

    elif next_action < next_img and next_action < next_mallet:
        _, _, _, _, cross_left, cross_right = sim.step(next_action)

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
err_dones = np.zeros((envs,), dtype=np.bool_)
rewards = np.zeros((2*envs,))
terminal_rewards = np.zeros((2*envs,))
close_side = np.ones((envs,), dtype=np.bool_)

past_xf = np.stack([obs[:envs, 28:30], obs[envs:, 28:30]], axis=1)
past_Vo = np.stack([obs[:envs, 30:32], obs[envs:, 30:32]], axis=1)

random_pos = np.random.uniform(np.array([mallet_r+0.05, width/2 - 0.15]), np.array([mallet_r+0.2, width/2+0.15]), (int(envs/2) - int(envs/4), 2))

xf_mallet_noise = np.empty((envs, 4))
block_reward = 0.4
actions_since_reset = 0 #np.zeros((envs,)) #np.random.randint(0, 60*59, (envs,))
env_idx_offset = 0

flipped = np.random.random((envs,)) > 0.5

action_map = np.array([
    [0.1, width/2 - goal_width/2 - 0.03, 0.8, 0],
    [0.15, width/2-goal_width/3, 0.8, 0],
    [0.17, width/2, 0.8, 0],
    [0.15, width/2 + goal_width/3, 0.8, 0],
    [0.1, width/2 + goal_width/3 + 0.03, 0.8, 0]])

q_action_idxs = np.full((int(envs/2),), 5, dtype=np.int32)

qpast_obs = np.empty((int(envs/2),obs_dim))
qacting = np.zeros((int(envs/2),), dtype=np.bool_)
qdones = np.zeros((int(envs/2),), dtype=np.bool_)
qrewards = np.zeros((int(envs/2),))
q_action = np.ones((int(envs/2),), dtype=np.int8)

if train:

    defender_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(400000),
        batch_size=batch_sizee
    )

    class TrajectoryBuffer:
        """Buffer that maintains trajectory structure for GAE computation"""
    
        def __init__(self, max_trajectory_length=1000):
            self.max_trajectory_length = max_trajectory_length
            self.trajectories = {}
            
        def add_step(self, obs, actions, rewards, dones, next_obs, log_probs, env_idx_offset=0):
            """Add a step to all environment trajectories"""
            batch_size = obs.shape[0]
            
            for env_idx in range(batch_size):
                if env_idx + env_idx_offset not in self.trajectories:
                    self.trajectories[env_idx+env_idx_offset] = {
                        'observations': [],
                        'actions': [],
                        'rewards': [],
                        'dones': [],
                        'next_observations': [],
                        'log_probs': []
                    }
                
                traj = self.trajectories[env_idx+env_idx_offset]
                traj['observations'].append(obs[env_idx].cpu())
                traj['actions'].append(actions[env_idx].cpu())
                traj['rewards'].append(rewards[env_idx].cpu())
                traj['dones'].append(dones[env_idx].cpu())
                traj['next_observations'].append(next_obs[env_idx].cpu())
                traj['log_probs'].append(log_probs[env_idx].cpu())
        
        def get_completed_trajectories(self, min_length=10):
            return_traj = []

            for env_idx, traj in self.trajectories.items():
                if len(traj['observations']) > 10:
                    trajectory_td = TensorDict({
                        'observation': torch.stack(traj['observations']),
                        'action': torch.stack(traj['actions']),
                        'next': TensorDict({
                            'reward': torch.stack(traj['rewards']),
                            'done': torch.stack(traj['dones']),
                            'observation': torch.stack(traj['next_observations']),
                            'terminated': torch.stack(traj['dones'])
                        }, batch_size=[len(traj['observations'])]),
                        'sample_log_prob': torch.stack(traj['log_probs'])
                    }, batch_size=[len(traj['observations'])])
                    
                    return_traj.append(trajectory_td)

            self.trajectories = {}
            return return_traj

        def clear_trajectories(self):
            self.trajectories = {}

    trajectory_buffer = TrajectoryBuffer(max_trajectory_length=1000)

    #Simulation loop
    for update in range(500000):  # Training loop
        print(update)
        print("simulating...")

        shooting = np.zeros((4,2,))

        for timestep in range(sim_steps):
            with torch.no_grad():
                if second_actor_filepath is None:
                    tensor_obs = TensorDict({"observation": torch.tensor(past_obs, dtype=torch.float32)}).to('cuda')

                    policy_out = policy_module(tensor_obs)
                    actions = policy_out["action"].detach().to('cpu')
                    policy_out["sample_log_prob"] = torch.maximum(policy_out["sample_log_prob"], torch.tensor(-8, dtype=torch.float32))
                    log_prob = policy_out["sample_log_prob"].detach().to('cpu')

                    actions_np = actions.numpy()
                else:

                    past_obs[envs:][flipped] = apply_symmetry(past_obs[envs:][flipped])

                    tensor_obs = TensorDict({"observation": torch.tensor(past_obs, dtype=torch.float32)}).to('cuda')

                    policy_out = policy_module(tensor_obs)
                    actions = policy_out["action"].detach().to('cpu')
                    policy_out["sample_log_prob"] = torch.maximum(policy_out["sample_log_prob"], torch.tensor(-8, dtype=torch.float32))
                    log_prob = policy_out["sample_log_prob"][:envs].detach().to('cpu')

                    actions_np = actions.numpy()                    

                    left_puck_fixed = past_obs[int(envs/4):int(envs/2),0] < bounds[0]/2

                    action_random_pos = np.concatenate([random_pos+0.01*(np.random.random((int(envs/2) - int(envs/4), 2))-0.5), np.full((int(envs/2) - int(envs/4), 2), np.array([0.8, 0]))], axis=1)

                    actions_np[envs+int(envs/4):envs+int(envs/2)][left_puck_fixed] = action_random_pos[left_puck_fixed]

                    actions_np[envs:, 1][flipped] = bounds[1] - actions_np[envs:, 1][flipped]
                    past_obs[envs:][flipped] = apply_symmetry(past_obs[envs:][flipped])

                    left_puck = past_obs[int(envs/2):envs,0] < bounds[0]/2

                    #tensor_obs = TensorDict({"observation": torch.tensor(past_obs[envs+int(envs/2):], dtype=torch.float32)}).to('cuda')

                    if left_puck.any():
                        if timestep % 5 == 0:
                            qrewards[:] = 0
                            qacting[:] = left_puck
                            qpast_obs[:] = past_obs[envs+int(envs/2):]
                            tensor_obs = TensorDict({"observation": torch.tensor(past_obs[envs+int(envs/2):][left_puck], dtype=torch.float32)}).to('cuda')
                            #tensor_obs = TensorDict({"observation": torch.tensor(past_obs[envs+int(envs/2):], dtype=torch.float32)}).to('cuda')

                            q_out = q_module(tensor_obs)
                            #print(q_out["action_value"][10])
                            q_values = q_out["action_value"]
                            probs = F.softmax(q_values, dim=-1)
                            # Sample an action from the categorical distribution
                            dist = Categorical(probs)
                            q_action[left_puck] = dist.sample().detach().cpu().numpy()
                            
                            q_action[left_puck] = np.where(np.random.rand(np.sum(left_puck)) < q_epsilon, np.random.randint(0,3, size=np.sum(left_puck)), q_action[left_puck])
                            q_action_idxs[left_puck] = np.clip(q_action[left_puck] - 1 + q_action_idxs[left_puck], 0, 4)
                        actions_np[envs+int(envs/2):, :][left_puck] = action_map[q_action_idxs[left_puck]]
                    
            xf = np.stack([actions_np[:envs, :2], actions_np[envs:,:2]], axis=1)
            xf[:,1,:] = bounds - xf[:,1,:]

            Vo = actions_np[:,2][:, None] * Vmax * np.stack((1+actions_np[:,3], 1-actions_np[:,3]), axis=1) + np.stack((np.random.normal(0, V_std_x, (2*envs)), np.random.normal(0, V_std_y, (2*envs))), axis=-1)
            too_low_voltage = np.logical_or(Vo[:,0] < 0.2, Vo[:,1] < 0.2)
            rewards[too_low_voltage] -= 0.1
            if too_low_voltage[0]:
                print("LOW VOLTAGE")
            Vo[:,0] = np.maximum(Vo[:,0], 0.03)
            Vo[:,1] = np.maximum(Vo[:,1], 0.03)
            Vo = np.stack([Vo[:envs,:], Vo[envs:,:]], axis=1)

            no_update_mask = (np.linalg.norm(past_obs[:,20:22] - past_obs[:,24:26], axis=1) < 0.01) & (np.linalg.norm(past_obs[:,28:30] - actions_np[:,:2], axis=1) < 0.01) & (np.linalg.norm(actions_np[:,:2] - past_obs[:,20:22], axis=1) < 0.01)
            #rewards[envs+int(envs/2):][no_update_mask[envs+int(envs/2):]] += 0.1
            no_update_mask = np.stack([no_update_mask[:envs], no_update_mask[envs:]], axis=1)

            xf[no_update_mask] = past_xf[no_update_mask]
            Vo[no_update_mask] = past_Vo[no_update_mask]

            past_xf = xf.copy()
            past_Vo = Vo.copy()

            sim.take_action(xf, Vo)
            #sim.impulse(2/60, 0.0034)

            obs[:, 24:28] = past_obs[:, 20:24]
            obs[:, 28:32] = actions_np

            rewards[:envs] += np.where((actions_np[:envs,0] < mallet_r+0.03) | (actions_np[:envs,0] > bounds[0]/2-mallet_r-0.03), -0.1, 0)
            rewards[:envs] += np.where((actions_np[:envs,1] < mallet_r+0.03) | (actions_np[:envs,1] > bounds[1] - mallet_r - 0.03), -0.1, 0)

            acc_time = sim.get_a()
            acc_time = np.max(acc_time[:,0,:],axis=1)
            apply_acctime = np.linalg.norm(obs[:envs,28:30] - obs[:envs,24:26], axis=1) > 0.2
            rewards[:envs][apply_acctime] += np.clip(np.arctan(20*np.clip(0.05 - acc_time[apply_acctime], -10, 0)) * 0.05, -0.1, 0)

            rewards[:envs] -= actions_np[:envs,2] * 0.003
            rewards[:envs] += np.where((actions_np[:envs,2] > 0.98) | (actions_np[:envs,2] < 0.02), -0.1, 0)
            rewards[:envs] += np.where((actions_np[:envs,3] > 0.98) | (actions_np[:envs,3] < -0.98), -0.1, 0)

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
                    mallet_pos, _, puck_pos, _, cross_left, cross_right = sim.step(next_img)

                    env_err = sim.check_state()
                    entered_left_goal_mask, entered_right_goal_mask = sim.check_goal()
                    if len(env_err) > 0:
                        print("err")
                        print(env_err)
                        for idx in env_err:
                            err_dones[idx] = True
                            if puck_pos[idx, 0] < bounds[0]/2:
                                terminal_rewards[idx] -= 100.0

                    terminal_rewards[:envs][entered_left_goal_mask] -= 100
                    err_dones[entered_left_goal_mask | entered_right_goal_mask] = True

                    terminal_rewards[:envs] += np.where(cross_right > 0, 20+np.sqrt(np.maximum(cross_right,0))/2, 0)
                    terminal_rewards[envs:] += np.where(cross_right > 0, -20.0, 0)

                    for i in range(4):
                        shooting[i,0] += np.sum(cross_right[int(envs*i/4):int(envs*(i+1)/4)] > 0)
                        shooting[i,1] += np.sum(cross_right[int(envs*i/4):int(envs*(i+1)/4)] != -1)

                    for i in range(2):
                        noise_idx = (puck_pos*100).astype(np.int16) + noise_seeds[:,i,:]
                        noise_idx[:,0] = np.clip(noise_idx[:,0], 0, 1999)
                        noise_idx[:,1] = np.clip(noise_idx[:,1], 0, 999)
                        puck_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]

                    beam_interference[:] = np.logical_and(puck_pos[:,0]+puck_r > beam_coeffs[0,0] * mallet_pos[:,0,0] + beam_coeffs[0,1], puck_pos[:,0]-puck_r < beam_coeffs[1,0] * mallet_pos[:,0,0] + beam_coeffs[1,1])

                    puck_wgn_noise = np.stack((np.random.normal(0, np.where(beam_interference[:envs], puck_beam_std[0,0] * puck_pos[:,0] + puck_beam_std[0,1], puck_std[0,0] * puck_pos[:,0] + puck_std[0,1]), (envs,)), np.random.normal(0, np.where(beam_interference[:envs], puck_beam_std[1,0] * puck_pos[:,0] + puck_beam_std[1,1], puck_std[1,0] * puck_pos[:,0] + puck_std[1,1]), (envs,))), axis=-1)
                    
                    large_beam_interference[:] = np.logical_and(puck_pos[:,0] > beam_coeffs[0,0] * mallet_pos[:,0,0] + beam_coeffs[0,1], puck_pos[:,0] < beam_coeffs[1,0] * mallet_pos[:,0,0] + beam_coeffs[1,1])

                    if large_beam_interference.any():
                        past_puck_data = camera_buffer.get([0])[:envs,:2] # player, x/y (envs,2)
                        miss_mask = np.random.random((envs)) < percent_miss[0] * puck_pos[:,0] + percent_miss[1] #a*x+b

                        puck_pos = np.concatenate([puck_pos + puck_noise + puck_wgn_noise, bounds - puck_pos],axis=0)
                        puck_pos[:envs][np.logical_and(large_beam_interference, miss_mask)] = past_puck_data[np.logical_and(large_beam_interference, miss_mask)]
                    else:
                        puck_pos = np.concatenate([puck_pos + puck_noise + puck_wgn_noise, bounds - puck_pos],axis=0)

                    #mallet_noise = np.empty((envs, 4)) #mallet, op mallet
                    for i in range(2,4):
                        noise_idx = (mallet_pos[:,int(i/2),:]*100).astype(np.int16) + noise_seeds[:,2+i%2,:]
                        noise_idx[:,0] = np.clip(noise_idx[:,0], 0, 1999)
                        noise_idx[:,1] = np.clip(noise_idx[:,1], 0, 999)
                        mallet_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]
                    op_mallet_pos = np.concatenate([mallet_pos[:,1,:] + mallet_noise[:,2:], bounds - mallet_pos[:,0,:]], axis=0)
                    camera_buffer.put(np.concatenate([puck_pos, op_mallet_pos], axis=1))

                    time_from_last_img = 0
                    agent_actions.subtract(next_img)
                    mallet_time.subtract(next_img)
                    agent_actions.put(np.clip(np.random.normal(image_delay[0], image_delay[1]), image_delay[2], image_delay[3]))
                    mallet_time.put(agent_actions.get(0) - np.clip(np.random.normal(mallet_delay[0], mallet_delay[1]), mallet_delay[2], mallet_delay[3]))

                    inference_img.put(np.logical_not(inference_img.get(0)))
                elif next_mallet < next_img and next_mallet < next_action:
                    mallet_pos, mallet_vel, _, _, cross_left, cross_right = sim.step(next_mallet)

                    env_err = sim.check_state()
                    entered_left_goal_mask, entered_right_goal_mask = sim.check_goal()
                    if len(env_err) > 0:
                        print("err")
                        print(env_err)
                        for idx in env_err:
                            err_dones[idx] = True
                            if puck_pos[idx, 0] < bounds[0]/2:
                                terminal_rewards[idx] -= 100.0

                    terminal_rewards[:envs][entered_left_goal_mask] -= 100
                    err_dones[entered_left_goal_mask | entered_right_goal_mask] = True

                    terminal_rewards[:envs] += np.where(cross_right > 0, 20+np.sqrt(np.maximum(cross_right,0))/2, 0)
                    terminal_rewards[envs:] += np.where(cross_right > 0, -20.0, 0)

                    for i in range(4):
                        shooting[i,0] += np.sum(cross_right[int(envs*i/4):int(envs*(i+1)/4)] > 0)
                        shooting[i,1] += np.sum(cross_right[int(envs*i/4):int(envs*(i+1)/4)] != -1)

                    #(2*envs, 2)
                    mal_noise = np.random.normal(0, mallet_std, (envs,2,2))
                    obs[:,20:22] = np.concatenate([mallet_pos[:,0,:] + mal_noise[:,0,:], bounds - mallet_pos[:,1,:] + mal_noise[:,1,:]], axis=0)
                    vel_noise = np.random.normal(0, mallet_vel_std, size=(envs,2,2))
                    obs[:,22:24] = np.concatenate([mallet_vel[:,0,:] + vel_noise[:,0,:], -mallet_vel[:,1,:] + vel_noise[:,1,:]], axis=0)
                    
                    time_from_last_img -= next_mallet
                    agent_actions.subtract(next_mallet)
                    mallet_time.subtract(next_mallet)

                elif next_action < next_img and next_action < next_mallet:
                    mallet_pos, _, puck_pos, puck_vel, cross_left, cross_right = sim.step(next_action)

                    env_err = sim.check_state()
                    entered_left_goal_mask, entered_right_goal_mask = sim.check_goal()
                    if len(env_err) > 0:
                        print("err")
                        print(env_err)
                        for idx in env_err:
                            err_dones[idx] = True
                            if puck_pos[idx, 0] < bounds[0]/2:
                                terminal_rewards[idx] -= 100.0

                    terminal_rewards[:envs][entered_left_goal_mask] -= 100
                    err_dones[entered_left_goal_mask | entered_right_goal_mask] = True

                    terminal_rewards[:envs] += np.where(cross_right > 0, 20+np.sqrt(np.maximum(cross_right,0))/2, 0)
                    terminal_rewards[envs:] += np.where(cross_right > 0, -20.0, 0)

                    for i in range(3):
                        shooting[i,0] += np.sum(cross_right[int(envs*i/4):int(envs*(i+1)/4)] > 0)
                        shooting[i,1] += np.sum(cross_right[int(envs*i/4):int(envs*(i+1)/4)] != -1)

                    camera_obs = camera_buffer.get(indices=[img_idx, img_idx+1, img_idx+2, img_idx+5, img_idx+11])
                    obs[:,:20] = camera_obs

                    camera_obs = camera_buffer.get(indices=[img_idx+3, img_idx+4, img_idx+5, img_idx+8, img_idx+14])
                    obs[envs+int(envs/2):,:20][obs[int(envs/2):envs,0] < bounds[0]/2] = camera_obs[envs+int(envs/2):][obs[int(envs/2):envs,0] < bounds[0]/2]

                    rewards[:envs] += np.where((puck_vel[:,0] == 0) & (puck_vel[:,1] == 0) & close_side, -0.05, 0)

                    time_from_last_img -= next_action
                    agent_actions.subtract(next_action)
                    mallet_time.subtract(next_action)

                    break

            actions_since_reset += 1

            rewards[:envs] -= np.linalg.norm(obs[:envs,20:22] - obs[:envs,24:26], axis=1) * 0.1

            dones[:envs] = (close_side & (obs[:envs,0] > bounds[0]/2)) | (err_dones)
            dones[envs:] = dones[:envs]      

            qdones = np.logical_and(left_puck, past_obs[int(envs/2):envs,0] > bounds[0]/2) | err_dones[int(envs/2):]

            rewards[dones] += terminal_rewards[dones]
            rewards /= 5
            terminal_rewards[dones] = 0

            flipped[dones[:envs]] = np.random.random((np.sum(dones[:envs]),)) > 0.5

            qrewards += rewards[envs+int(envs/2):]   

            #print("---")
            #print(rewards[envs + int(envs/2) + 10])
            #print(terminal_rewards)
            #print(dones[envs+int(envs/2)+10])
            #print(value_module(tensor_obs)["state_value"][0])
            #print(past_obs[0])
            #print(obs[0,28:32])
            #print(apply_symmetry(torch.tensor(past_obs[0])[None, :]))

            if second_actor_filepath is None:
                trajectory_buffer.add_step(
                        obs=torch.tensor(past_obs, dtype=torch.float32),
                        actions=torch.tensor(obs[:, 28:32], dtype=torch.float32),
                        rewards=torch.tensor(rewards, dtype=torch.float32),
                        dones=torch.tensor(dones, dtype=torch.bool),
                        next_obs=torch.tensor(obs, dtype=torch.float32),
                        log_probs=log_prob,
                        env_idx_offset=env_idx_offset
                    )
            else:
                trajectory_buffer.add_step(
                        obs=torch.tensor(past_obs[:envs], dtype=torch.float32),
                        actions=torch.tensor(obs[:envs, 28:32], dtype=torch.float32),
                        rewards=torch.tensor(rewards[:envs], dtype=torch.float32),
                        dones=torch.tensor(dones[:envs], dtype=torch.bool),
                        next_obs=torch.tensor(obs[:envs], dtype=torch.float32),
                        log_probs=log_prob,
                        env_idx_offset=env_idx_offset
                    )

                defender_step = qacting & ((timestep % 5 == 4) | qdones)

                if defender_step.any():
                    td = TensorDict({
                            "observation": torch.tensor(qpast_obs[defender_step], dtype=torch.float32),
                            "action": torch.tensor(q_action[defender_step][:,None], dtype=torch.int64),
                            "next": TensorDict({
                                "reward": torch.tensor(qrewards[defender_step][:,None], dtype=torch.float32),
                                "observation": torch.tensor(obs[envs+int(envs/2):][defender_step], dtype=torch.float32),
                                "done": torch.tensor(qdones[defender_step][:,None], dtype=torch.bool),
                            }, batch_size=[defender_step.sum()]),
                        }, batch_size=[defender_step.sum()])

                    for i in range(td.batch_size[0]):
                        single_transition = td[i]
                        defender_buffer.add(single_transition)

            qacting[qdones] = False
                    
            sim.display_state(envs-10)

            #print("---")
            #print(past_obs[0])
            #print(actions[0])
            #print(rewards[0])
            #print(dones[0])
            #print(obs[0])
            #print(log_prob[0])
            #input()

            dones[:envs] = err_dones
            dones[envs:] = err_dones

            if actions_since_reset == 150:
                actions_since_reset = 0
                random_pos = np.random.uniform(np.array([mallet_r+0.05, width/2 - 0.15]), np.array([mallet_r+0.2, width/2+0.15]), (int(envs/2) - int(envs/4), 2))
                flipped = np.random.random((envs,)) > 0.5
                dones[:] = True
                env_idx_offset += envs

            num_resets = int(np.sum(dones) / 2)

            if num_resets > 0:
                ab_vars = np.random.uniform(
                        low=ab_ranges[:, 0].reshape(1, 1, 6),    # Shape: (1, 1, 5)
                        high=ab_ranges[:, 1].reshape(1, 1, 6),   # Shape: (1, 1, 5)
                        size=(num_resets, 2, 6)                     # Output shape
                    )

                ab_vars *= np.random.uniform(speed_var[0], speed_var[1], size=(num_resets, 2, 1))

                ab_obs = np.zeros((2*num_resets, 6))
                    
                ab_obs[:num_resets, :] = (ab_vars[:,0,:] / pullyR) * ab_obs_scaling[:]
                ab_obs[num_resets:, :] = (ab_vars[:,1,:] / pullyR) * ab_obs_scaling[:]

                obs[dones] = np.concatenate([obs_init[dones], ab_obs], axis=1)

                camera_buffer.reset(np.where(dones)[0])

                sim.reset_sim(np.where(dones[:envs])[0], ab_vars)

            past_obs = obs.copy()
            dones[:] = False
            rewards[:] = 0
            err_dones[:] = False
            close_side = obs[:envs,0] < bounds[0]/2

            if env_idx_offset == 2*envs:
                break
                
        print("training...")
        actions_since_reset = 0
        env_idx_offset = 0
        torch.cuda.empty_cache()

        trajectories = trajectory_buffer.get_completed_trajectories()

        if trajectories is None:
            continue
        
        all_processed_data = []

        for trajectory_td in trajectories:
            #if torch.sum(trajectory_td['next', 'reward']) < -200:
            #    print(trajectory_td['next', 'reward'])
            trajectory_td = trajectory_td.to('cuda')

            # Compute GAE advantages and returns
            with torch.no_grad():
                gae_module(trajectory_td)
            
            # GAE adds 'advantage' and 'value_target' to the tensordict
            # Flatten trajectory for training (convert from [seq_len] to [seq_len] individual samples)
            seq_len = trajectory_td.batch_size[0]

            trajectory_td = trajectory_td.to('cpu')
            
            for t in range(seq_len):
                sample_td = TensorDict({
                    'observation': trajectory_td['observation'][t],
                    'action': trajectory_td['action'][t],
                    'next': TensorDict({
                        'reward': trajectory_td['next']['reward'][t],
                        'done': trajectory_td['next']['done'][t],
                        'observation': trajectory_td['next']['observation'][t],
                        'terminated': trajectory_td['next']['terminated'][t]
                    }, batch_size=[]),
                    'sample_log_prob': trajectory_td['sample_log_prob'][t],
                    'advantage': trajectory_td['advantage'][t],
                    'value_target': trajectory_td['value_target'][t]
                }, batch_size=[])
                
                all_processed_data.append(sample_td)

            del trajectory_td
            torch.cuda.empty_cache()
        
        if len(all_processed_data) == 0:
            continue
        
        # Stack all samples for batch training
        batch_td = torch.stack(all_processed_data)
        del all_processed_data
        torch.cuda.empty_cache()

        train_data = batch_td.to('cuda')

        del batch_td

        for _ in range(num_epochs):
            perm = torch.randperm(train_data.batch_size[0])
            train_data = train_data[perm]
            del perm

            batch_size = min(batch_sizee, train_data.batch_size[0])
            num_batches = (train_data.batch_size[0] + batch_size - 1) // batch_size

            losses = np.empty((num_batches-1,3))

            for i in range(num_batches-1):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, train_data.batch_size[0])
                mini_batch = train_data[start_idx:end_idx] #.to('cuda')

                loss_vals = loss_module(mini_batch)
                loss = loss_vals["loss_critic"] + loss_vals["loss_objective"] +  loss_vals["loss_entropy"]

                losses[i,0] = loss_vals["loss_objective"].detach().cpu().numpy()
                losses[i,1] = loss_vals["loss_critic"].detach().cpu().numpy()
                losses[i,2] = loss_vals["loss_entropy"].detach().cpu().numpy()

                optimizer_value.zero_grad()
                optimizer_policy.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_module.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(value_module.parameters(), 0.5)
              
                #if loss_vals["kl_approx"] < 1.5*target_kl:
                optimizer_policy.step()

                optimizer_value.step()

                del mini_batch
                torch.cuda.empty_cache()

            print(losses[:,0].mean())
            print(losses[:,1].mean())
            print(losses[:,2].mean())
            print((losses[:,0] + losses[:,1] + losses[:,2]).mean())

        del train_data

        defender_losses = np.empty((int(len(defender_buffer)/batch_sizee),))

        for i in range(int(len(defender_buffer)/batch_sizee)):
            batch = defender_buffer.sample().to('cuda')
            loss = q_loss_module(batch)["loss"]
            defender_losses[i] = loss
            optimizer_defender.zero_grad()
            loss.backward()
            optimizer_defender.step()
            torch.cuda.empty_cache()

        print(defender_losses.mean())    

        print(f"Episode {update}")
        #print(batch_td["next", "reward"].mean().item())
        #print(f"Training on {batch_td.batch_size[0]} samples (including augmented data)")
        for i in range(4):
            print(shooting[i,0] / shooting[i,1])
       
        #if update % 25 == 0:
        torch.save({
            'policy_state_dict': policy_module.state_dict(),
            'value_state_dict': value_module.state_dict(),
            'q_state_dict': q_module.state_dict(),
            'policy_optim_state_dict': optimizer_policy.state_dict(),
            'value_optim_state_dict': optimizer_value.state_dict(),
            'q_optim_state_dict': optimizer_defender.state_dict(),
            #'optim_state_dict': optimizer.state_dict(),
        }, save_filepath)

random_pos = np.random.uniform(np.array([mallet_r+0.05, mallet_r+0.05]), np.array([bounds[0]/2-mallet_r-0.05, bounds[1] - mallet_r-0.05]), (1, 2))
q_action_idxs = np.full((1,), 5, dtype=np.int32)

shooting = np.zeros((4,2,))
update = 0
for timestep in range(100000):
    with torch.no_grad():
        if second_actor_filepath is None:
            tensor_obs = TensorDict({"observation": torch.tensor(past_obs, dtype=torch.float32)}).to('cuda')

            policy_out = policy_module(tensor_obs)
            actions = policy_out["action"].detach().to('cpu')
            policy_out["sample_log_prob"] = torch.maximum(policy_out["sample_log_prob"], torch.tensor(-8, dtype=torch.float32))
            log_prob = policy_out["sample_log_prob"].detach().to('cpu')

            actions_np = actions.numpy()
        else:

            past_obs[envs:][flipped] = apply_symmetry(past_obs[envs:][flipped])

            tensor_obs = TensorDict({"observation": torch.tensor(past_obs, dtype=torch.float32)}).to('cuda')

            policy_out = policy_module(tensor_obs)
            actions = policy_out["action"].detach().to('cpu')
            policy_out["sample_log_prob"] = torch.maximum(policy_out["sample_log_prob"], torch.tensor(-8, dtype=torch.float32))
            log_prob = policy_out["sample_log_prob"][:envs].detach().to('cpu')

            actions_np = actions.numpy()                    

            left_puck_fixed = obs[int(envs/4):int(envs/2),0] < bounds[0]/2

            #action_random_pos = np.concatenate([random_pos+0.01*(np.random.random((int(envs/2) - int(envs/4), 2))-0.5), np.full((int(envs/2) - int(envs/4), 2), np.array([0.8, 0]))], axis=1)

            #actions_np[envs+int(envs/4):envs+int(envs/2)][left_puck_fixed] = action_random_pos[left_puck_fixed]

            actions_np[envs:, 1][flipped] = bounds[1] - actions_np[envs:, 1][flipped]
            past_obs[envs:][flipped] = apply_symmetry(past_obs[envs:][flipped])

            left_puck = past_obs[int(envs/2):envs,0] < bounds[0]/2

            if left_puck.any():
                if timestep % 5 == 0:
                    tensor_obs = TensorDict({"observation": torch.tensor(past_obs[envs+int(envs/2):][left_puck], dtype=torch.float32)}).to('cuda')
                    #tensor_obs = TensorDict({"observation": torch.tensor(past_obs[envs+int(envs/2):], dtype=torch.float32)}).to('cuda')

                    q_out = q_module(tensor_obs)
                    #print(q_out["action_value"][10])
                    q_values = q_out["action_value"]
                    probs = F.softmax(q_values, dim=-1)

                    # Sample an action from the categorical distribution
                    dist = Categorical(probs)
                    q_action = dist.sample().detach().cpu().numpy()
                    
                    q_action = np.where(np.random.rand(np.sum(left_puck)) < q_epsilon, np.random.randint(0,3, size=np.sum(left_puck)), q_action)
                    q_action_idxs[left_puck] = np.clip(q_action - 1 + q_action_idxs[left_puck], 0, 4)
                actions_np[envs+int(envs/2):, :][left_puck] = action_map[q_action_idxs[left_puck]]
            
    xf = np.stack([actions_np[:envs, :2], actions_np[envs:,:2]], axis=1)
    xf[:,1,:] = bounds - xf[:,1,:]

    Vo = actions_np[:,2][:, None] * Vmax * np.stack((1+actions_np[:,3], 1-actions_np[:,3]), axis=1) + np.stack((np.random.normal(0, V_std_x, (2*envs)), np.random.normal(0, V_std_y, (2*envs))), axis=-1)
    too_low_voltage = np.logical_or(Vo[:,0] < 0.3, Vo[:,1] < 0.3)
    rewards[too_low_voltage] -= 0.1
    if too_low_voltage[0]:
        print("LOW VOLTAGE")
    Vo[:,0] = np.maximum(Vo[:,0], 0.03)
    Vo[:,1] = np.maximum(Vo[:,1], 0.03)
    Vo = np.stack([Vo[:envs,:], Vo[envs:,:]], axis=1)

    no_update_mask = (np.linalg.norm(past_obs[:,20:22] - past_obs[:,24:26], axis=1) < 0.01) & (np.linalg.norm(past_obs[:,28:30] - actions_np[:,:2], axis=1) < 0.01) & (np.linalg.norm(actions_np[:,:2] - past_obs[:,20:22], axis=1) < 0.01)
    #rewards[envs+int(envs/2):][no_update_mask[envs+int(envs/2):]] += 0.1
    no_update_mask = np.stack([no_update_mask[:envs], no_update_mask[envs:]], axis=1)

    xf[no_update_mask] = past_xf[no_update_mask]
    Vo[no_update_mask] = past_Vo[no_update_mask]

    past_xf = xf.copy()
    past_Vo = Vo.copy()

    sim.take_action(xf, Vo)
    sim.impulse(2/60, 0.0034)

    obs[:, 24:28] = past_obs[:, 20:24]
    obs[:, 28:32] = actions_np

    rewards[:envs] += np.where((actions_np[:envs,0] < mallet_r+0.03) | (actions_np[:envs,0] > bounds[0]/2-mallet_r-0.03), -0.1, 0)
    rewards[:envs] += np.where((actions_np[:envs,1] < mallet_r+0.03) | (actions_np[:envs,1] > bounds[1] - mallet_r - 0.03), -0.1, 0)

    rewards[:envs] -= actions_np[:envs,2] * 0.003
    rewards[:envs] += np.where((actions_np[:envs,2] > 0.98) | (actions_np[:envs,2] < 0.02), -0.1, 0)
    rewards[:envs] += np.where((actions_np[:envs,3] > 0.98) | (actions_np[:envs,3] < -0.98), -0.1, 0)

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
            mallet_pos, _, puck_pos, _, cross_left, cross_right = sim.step(next_img)

            env_err = sim.check_state()
            entered_left_goal_mask, entered_right_goal_mask = sim.check_goal()
            if len(env_err) > 0:
                print("err")
                print(env_err)
                for idx in env_err:
                    err_dones[idx] = True
                    if puck_pos[idx, 0] < bounds[0]/2:
                        terminal_rewards[idx] -= 100.0

            terminal_rewards[:envs][entered_left_goal_mask] -= 100
            err_dones[entered_left_goal_mask | entered_right_goal_mask] = True

            terminal_rewards[:envs] += np.where(cross_right > 0, 20+np.sqrt(np.maximum(cross_right,0))/2, 0)
            terminal_rewards[envs:] += np.where(cross_right > 0, -20.0, 0)

            for i in range(4):
                shooting[i,0] += np.sum(cross_right[int(envs*i/4):int(envs*(i+1)/4)] > 0)
                shooting[i,1] += np.sum(cross_right[int(envs*i/4):int(envs*(i+1)/4)] != -1)

            for i in range(2):
                noise_idx = (puck_pos*100).astype(np.int16) + noise_seeds[:,i,:]
                noise_idx[:,0] = np.clip(noise_idx[:,0], 0, 1999)
                noise_idx[:,1] = np.clip(noise_idx[:,1], 0, 999)
                puck_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]

            beam_interference[:] = np.logical_and(puck_pos[:,0]+puck_r > beam_coeffs[0,0] * mallet_pos[:,0,0] + beam_coeffs[0,1], puck_pos[:,0]-puck_r < beam_coeffs[1,0] * mallet_pos[:,0,0] + beam_coeffs[1,1])

            puck_wgn_noise = np.stack((np.random.normal(0, np.where(beam_interference[:envs], puck_beam_std[0,0] * puck_pos[:,0] + puck_beam_std[0,1], puck_std[0,0] * puck_pos[:,0] + puck_std[0,1]), (envs,)), np.random.normal(0, np.where(beam_interference[:envs], puck_beam_std[1,0] * puck_pos[:,0] + puck_beam_std[1,1], puck_std[1,0] * puck_pos[:,0] + puck_std[1,1]), (envs,))), axis=-1)
            
            large_beam_interference[:] = np.logical_and(puck_pos[:,0] > beam_coeffs[0,0] * mallet_pos[:,0,0] + beam_coeffs[0,1], puck_pos[:,0] < beam_coeffs[1,0] * mallet_pos[:,0,0] + beam_coeffs[1,1])

            if large_beam_interference.any():
                past_puck_data = camera_buffer.get([0])[:envs,:2] # player, x/y (envs,2)
                miss_mask = np.random.random((envs)) < percent_miss[0] * puck_pos[:,0] + percent_miss[1] #a*x+b

                puck_pos = np.concatenate([puck_pos + puck_noise + puck_wgn_noise, bounds - puck_pos],axis=0)
                puck_pos[:envs][np.logical_and(large_beam_interference, miss_mask)] = past_puck_data[np.logical_and(large_beam_interference, miss_mask)]
            else:
                puck_pos = np.concatenate([puck_pos + puck_noise + puck_wgn_noise, bounds - puck_pos],axis=0)

            #mallet_noise = np.empty((envs, 4)) #mallet, op mallet
            for i in range(2, 4):
                noise_idx = (mallet_pos[:,int(i/2),:]*100).astype(np.int16) + noise_seeds[:,2+i%2,:]
                noise_idx[:,0] = np.clip(noise_idx[:,0], 0, 1999)
                noise_idx[:,1] = np.clip(noise_idx[:,1], 0, 999)
                mallet_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]
            mallet_wgn_noise = np.stack((np.random.normal(0, puck_std[0,0] * mallet_pos[:,1,0] + puck_std[0,1], (envs,)), np.random.normal(0, puck_std[1,0] * mallet_pos[:,1,0] + puck_std[1,1], (envs,))), axis=-1)

            op_mallet_pos = np.concatenate([mallet_pos[:,1,:] + mallet_noise[:,2:] + mallet_wgn_noise, bounds - mallet_pos[:,0,:]], axis=0)

            camera_buffer.put(np.concatenate([puck_pos, op_mallet_pos], axis=1))

            time_from_last_img = 0
            agent_actions.subtract(next_img)
            mallet_time.subtract(next_img)
            agent_actions.put(np.clip(np.random.normal(image_delay[0], image_delay[1]), image_delay[2], image_delay[3]))
            mallet_time.put(agent_actions.get(0) - np.clip(np.random.normal(mallet_delay[0], mallet_delay[1]), mallet_delay[2], mallet_delay[3]))

            inference_img.put(np.logical_not(inference_img.get(0)))
        elif next_mallet < next_img and next_mallet < next_action:
            mallet_pos, mallet_vel, _, _, cross_left, cross_right = sim.step(next_mallet)

            env_err = sim.check_state()
            entered_left_goal_mask, entered_right_goal_mask = sim.check_goal()
            if len(env_err) > 0:
                print("err")
                print(env_err)
                for idx in env_err:
                    err_dones[idx] = True
                    if puck_pos[idx, 0] < bounds[0]/2:
                        terminal_rewards[idx] -= 100.0

            terminal_rewards[:envs][entered_left_goal_mask] -= 100
            err_dones[entered_left_goal_mask | entered_right_goal_mask] = True

            terminal_rewards[:envs] += np.where(cross_right > 0, 20+np.sqrt(np.maximum(cross_right,0))/2, 0)
            terminal_rewards[envs:] += np.where(cross_right > 0, -20.0, 0)

            for i in range(4):
                shooting[i,0] += np.sum(cross_right[int(envs*i/4):int(envs*(i+1)/4)] > 0)
                shooting[i,1] += np.sum(cross_right[int(envs*i/4):int(envs*(i+1)/4)] != -1)

            #(2*envs, 2)
            mal_noise = np.random.normal(0, mallet_std, (envs,2,2))
            obs[:,20:22] = np.concatenate([mallet_pos[:,0,:] + mal_noise[:,0,:], bounds - mallet_pos[:,1,:] + mal_noise[:,1,:]], axis=0)
            vel_noise = np.random.normal(0, mallet_vel_std, size=(envs,2,2))
            obs[:,22:24] = np.concatenate([mallet_vel[:,0,:] + vel_noise[:,0,:], -mallet_vel[:,1,:] + vel_noise[:,1,:]], axis=0)
            
            time_from_last_img -= next_mallet
            agent_actions.subtract(next_mallet)
            mallet_time.subtract(next_mallet)

        elif next_action < next_img and next_action < next_mallet:
            mallet_pos, _, puck_pos, puck_vel, cross_left, cross_right = sim.step(next_action)

            env_err = sim.check_state()
            entered_left_goal_mask, entered_right_goal_mask = sim.check_goal()
            if len(env_err) > 0:
                print("err")
                print(env_err)
                for idx in env_err:
                    err_dones[idx] = True
                    if puck_pos[idx, 0] < bounds[0]/2:
                        terminal_rewards[idx] -= 100.0

            terminal_rewards[:envs][entered_left_goal_mask] -= 100
            err_dones[entered_left_goal_mask | entered_right_goal_mask] = True

            terminal_rewards[:envs] += np.where(cross_right > 0, 20+np.sqrt(np.maximum(cross_right,0))/2, 0)
            terminal_rewards[envs:] += np.where(cross_right > 0, -20.0, 0)

            for i in range(3):
                shooting[i,0] += np.sum(cross_right[int(envs*i/4):int(envs*(i+1)/4)] > 0)
                shooting[i,1] += np.sum(cross_right[int(envs*i/4):int(envs*(i+1)/4)] != -1)

            camera_obs = camera_buffer.get(indices=[img_idx, img_idx+1, img_idx+2, img_idx+5, img_idx+11])
            obs[:,:20] = camera_obs

            camera_obs = camera_buffer.get(indices=[img_idx+3, img_idx+4, img_idx+5, img_idx+8, img_idx+14])
            obs[envs+int(envs/2):,:20][obs[int(envs/2):envs,0] < bounds[0]/2] = camera_obs[envs+int(envs/2):][obs[int(envs/2):envs,0] < bounds[0]/2]

            rewards[:envs] += np.where((puck_vel[:,0] == 0) & (puck_vel[:,1] == 0) & close_side, -0.05, 0)

            time_from_last_img -= next_action
            agent_actions.subtract(next_action)
            mallet_time.subtract(next_action)

            break

    actions_since_reset += 1

    rewards[:envs] -= np.linalg.norm(obs[:envs,20:22] - obs[:envs,24:26], axis=1) * 0.1

    dones[:envs] = (close_side & (obs[:envs,0] > bounds[0]/2)) | (err_dones)
    dones[envs:] = dones[:envs]            

    rewards[dones] += terminal_rewards[dones]
    rewards /= 5
    terminal_rewards[dones] = 0

    flipped[dones[:envs]] = np.random.random((np.sum(dones[:envs]),)) > 0.5

    #print("---")
    #print(rewards[envs + int(envs/2) + 10])
    #print(terminal_rewards)
    #print(dones[envs+int(envs/2)+10])
    #print(value_module(tensor_obs)["state_value"][0])
    #print(past_obs[0])
    #print(obs[0,28:32])
    #print(apply_symmetry(torch.tensor(past_obs[0])[None, :]))
            
    sim.display_state(0)

    #print("---")
    #print(past_obs[0])
    #print(actions[0])
    #print(rewards[0])
    #print(dones[0])
    #print(obs[0])
    #print(log_prob[0])
    #input()

    dones[:envs] = err_dones
    dones[envs:] = err_dones

    num_resets = int(np.sum(dones) / 2)

    if num_resets > 0:
        
        obs[dones] = np.concatenate([obs_init[dones], ab_obs], axis=1)

        camera_buffer.reset(np.where(dones)[0])

        sim.reset_sim(np.where(dones[:envs])[0], ab_vars)

    past_obs = obs.copy()
    dones[:] = False
    rewards[:] = 0
    err_dones[:] = False
    close_side = obs[:envs,0] < bounds[0]/2
