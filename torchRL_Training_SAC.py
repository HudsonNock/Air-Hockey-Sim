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
import torch.nn.functional as F4
import copy
from torchrl.data import PrioritizedReplayBuffer, LazyTensorStorage

load_filepath = "checkpoints\\SAC_173.pth"
save_filepath = "checkpoints\\SAC_174.pth" #Best is 12, 15, 22, 44, 76, 103, 121, 134, 142, 160, 165, 178
train = True
# Simulation parameters
obs_dim = 39 #[puck_pos, opp_mallet_pos] #guess mean low stdl
            #[mallet_pos, mallet_vel, past mallet_pos, past malet_vel, past action]
            #[a1, a2, a3, b1, b2, b3]
            # opp defending
            # 20 + 8 + 4 + 6 + 1 = 39

mallet_r = 0.1010 / 2
puck_r = 0.0826 / 2
pullyR = 0.0357347133

#col: n_f + (1-n_f/n_0) * 2/(1+e^(n_r x^2)) n_0

action_dim = 5
Vmax = 24*0.8

puck_std = np.array([[0.000049,0.000007], [0.000024,0.000035]]) # std in x y, a*x + b
puck_beam_std = np.array([[-0.000169,0.000926], [-0.000072,0.000662]]) #std in x y, a*x+b
percent_miss = np.array([0.2, 0.25]) #a*x+b
puck_perlin_std = 0.0012

camera_pos = [0.4, 1.5] #x,z
beam_width = 0.0508
beam_thickness = 0.0254
beam_height = 0.06765 #from floor roof of beam

mallet_std = 0.0005
mallet_vel_std = 0.07

V_std_x = 0.07
V_std_y = 0.07

# (7.629e-06, 6.617e-03, 7e-02, -7.445e-06, -2.661e-03, 5.277e-03

ab_ranges = np.array([
        [1.8e-05, 2.3e-05], #[6.9e-6, 8.5e-6],  # a1
        [9.3e-3, 1.10e-2],  # a2
        [5.25e-2, 8.15e-2],  # a3
        [-1.4e-05, -1.1e-05], #[-8.19e-6, -6.01e-6], # b1
        [-5.5e-3, -4.6e-3],   # b2
	[3.5e-4, 5.5e-4]    #[3.5e-3, 6.5e-3] #b3
    ])

#[2.259685e-05, 1.0497799e-02, 6.2993579e-02, -1.256968e-05, -5.087524e-03, 4.398169e-04])

speed_var = [0.7,1.2]

#mean, std, min, max
image_delay = [15.166/1000, 0.3/1000, 14/1000, 16.5/1000]
mallet_delay = [7.17/1000, 0.3/1000, 6.2/1000, 8.0/1000]
camera_period = 1/120.0

"""
Mallet Delay

4.6-4.7ms: 16 frames (3.2%)
5.3-5.4ms: 166 frames (33.5%)
6.1-6.2ms: 158 frames (31.9%)
6.8-6.9ms: 144 frames (29.1%)
7.6-7.7ms: 2 frames (0.4%)
"""

frames = [0, 1, 2, 5, 11]

height = 2.362
width = 1.144
bounds = np.array([height, width])
goal_width = 0.3345

lr_val = 5e-4
policy_lrs = [lr_val, lr_val, lr_val] #policy, def1, def2
Q_lrs = [lr_val, lr_val, lr_val] #policy, def1, def2

gamma = 0.99
tau = 0.005

batch_size = 1024
img_offset = 12 #17

#q, pol, dec

#------------------

beam_coeffs = np.array([[camera_pos[1]/(camera_pos[1]-beam_height+beam_thickness), (beam_height-beam_thickness)*(camera_pos[0]-beam_width/2)/(camera_pos[1]-beam_height+beam_thickness) - beam_width/2], [1+(beam_height)/(camera_pos[1]-beam_height), beam_height*(camera_pos[0]+beam_width/2)/(camera_pos[1]-beam_height) + beam_width/2]])

policy_modules = [] #[policy, def1, def2]
for _ in range(2):
    policy_modules.append(nn.Sequential(
                            nn.Linear(obs_dim, 512),
                            nn.LayerNorm(512),
                            nn.LeakyReLU(0.01),
                            nn.Linear(512, 256),
                            nn.LayerNorm(256),
                            nn.LeakyReLU(0.01),
                            nn.Linear(256, 128),
                            nn.LeakyReLU(0.01),
                            nn.Linear(128, 64),
                            nn.LeakyReLU(),
                            nn.Linear(64, action_dim * 2),
                            NormalParamExtractor()
                        ))

#Q nets
Q_modules = [] # [policy 1/2, def1 1/2, def2 1/2]
for _ in range(4):
    Q_modules.append(nn.Sequential(
                        nn.Linear(obs_dim + action_dim, 512),
                        nn.LayerNorm(512),
                        nn.LeakyReLU(0.01),
                        nn.Linear(512, 256),
                        nn.LayerNorm(256),
                        nn.LeakyReLU(0.01),
                        nn.Linear(256, 128),
                        nn.LeakyReLU(0.01),
                        nn.Linear(128, 64),
                        nn.LeakyReLU(),
                        nn.Linear(64, 1)
                    ))

action_center = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)

def init_policy(m):
    if isinstance(m, nn.Linear):
        if m.out_features == action_dim * 2:
            # Initialize weights to 0 so output = bias
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
            m.weight.data *= 0.03
            # Bias: loc = center, log_std = 0
            bias = torch.cat([action_center, torch.ones(action_dim)*0.5], dim=0)
            with torch.no_grad():
                m.bias.copy_(bias)
        elif m.out_features == 1:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
            m.weight.data *= 0.03
            with torch.no_grad():
                m.bias.copy_(torch.zeros((1,)))
        else:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
            nn.init.zeros_(m.bias)

for pol in policy_modules:
    pol.apply(init_policy)

for q in Q_modules:
    q.apply(init_policy)

for i in range(2):
    policy_modules[i] = TensorDictModule(
                            policy_modules[i], in_keys=["observation"], out_keys=["loc", "scale"]
                        ).to('cuda')

class QNetWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, observation, action):
        # Concatenate features and actions along the last dimension
        combined = torch.cat([observation, action], dim=-1)
        return self.net(combined)

# Wrap your existing nets
for i in range(4):
    Q_modules[i] = TensorDictModule(
                        QNetWrapper(Q_modules[i]), 
                        in_keys=["observation", "action"], 
                        out_keys=["q_value"]
                    ).to('cuda')
    Q_modules[i] = TensorDictModule(
                        Q_modules[i], in_keys=["observation", "action"], out_keys=["q_value"]
                    ).to('cuda')

low = torch.tensor([mallet_r+0.01, mallet_r+0.01, 0, -1, 0], dtype=torch.float32).to('cuda')
high = torch.tensor([bounds[0]/2-mallet_r-0.01, bounds[1] - mallet_r-0.01, 1, 1, 1], dtype=torch.float32).to('cuda')

lows = [[mallet_r+0.01, mallet_r+0.01, 0, -1, 0],
        [mallet_r+0.01, mallet_r+0.01, 0, -1, 0],
        [mallet_r+0.01, mallet_r+0.11, 0, -1, 0]]

highs = [[bounds[0]/2-mallet_r-0.01, bounds[1] - mallet_r-0.01, 1, 1, 1],
            [bounds[0]/2-mallet_r-0.01-2*puck_r, bounds[1] - mallet_r-0.01, 0.7, 1, 1],
            [mallet_r+0.05, bounds[1] - mallet_r-0.11, 0.5, 1, 1]]

for i in range(2):
    if train:
        policy_modules[i] = ProbabilisticActor(
            module=policy_modules[i],
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": torch.tensor(lows[i], dtype=torch.float32),
                "high": torch.tensor(highs[i], dtype=torch.float32), 
            },
            default_interaction_type=tensordict.nn.InteractionType.RANDOM,
            return_log_prob=True,
        ).to('cuda')
    else:
        policy_modules[i] = ProbabilisticActor( 
            module=policy_modules[i],
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": torch.tensor(lows[i], dtype=torch.float32),
                "high": torch.tensor(highs[i], dtype=torch.float32), 
            },
            default_interaction_type=tensordict.nn.InteractionType.RANDOM,
            return_log_prob=True,
        ).to('cuda')

target_entropy = [-float(3.5), -float(3.6)]


log_alphas = []
for j in range(2):
    if j == 0:
        log_alphas.append(torch.full((1,), -5.5, requires_grad=True, device='cuda'))
    else:
        log_alphas.append(torch.full((1,), -5.5, requires_grad=True, device='cuda'))

alpha_optimizers = []
for i in range(2):
    alpha_optimizers.append(torch.optim.Adam([log_alphas[i]], lr=1e-4))

policy_optimizers = []
for i in range(2):
    policy_optimizers.append(torch.optim.Adam(policy_modules[i].parameters(), lr=policy_lrs[i]))

Q_optimizers = []
for i in range(2):
    Q_optimizers.append(torch.optim.Adam(list(Q_modules[2*i].parameters()) + list(Q_modules[2*i+1].parameters()), lr=Q_lrs[i]))

Q_modules_target = []
for i in range(4):
    Q_modules_target.append(copy.deepcopy(Q_modules[i]).to('cuda'))

class ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

#if load_filepath is not None:
#    checkpoint = torch.load(load_filepath, map_location='cuda')
#    policy_module.load_state_dict(checkpoint['policy_state_dict'])
#    value_module.load_state_dict(checkpoint['value_state_dict'])
#    
#    optimizer_policy.load_state_dict(checkpoint['policy_optim_state_dict'])
#    optimizer_value.load_state_dict(checkpoint['value_optim_state_dict'])

def update_sac_agent(i, batch_size, gamma=0.99, tau=0.005):
    # Select components for agent i
    if i == 0:
        buffer = policy_buffer
    elif i == 1:
        buffer = def_buffer

    actor = policy_modules[i]
    q1, q2 = Q_modules[2*i], Q_modules[2*i+1]
    t_q1, t_q2 = Q_modules_target[2*i], Q_modules_target[2*i+1]
    log_alpha = log_alphas[i]
    
    batch, info = buffer.sample(batch_size, return_info=True)
    batch = batch.to('cuda')
    next_batch = batch.get("next")
    
    # --- 1. Critic Update ---
    with torch.no_grad():
        actor(next_batch)
        
        next_log_prob = torch.clamp(next_batch.get("sample_log_prob"), -7, 100)
        #gate_value = next_batch.get("action")[:, 4:5]
        #entropy_mask = (1-gate_value)
        
        target_q1 = t_q1(next_batch).get("q_value")
        target_q2 = t_q2(next_batch).get("q_value")
        
        alpha = log_alpha.exp()
        target_q = torch.min(target_q1, target_q2) - (alpha * next_log_prob)[:,None]
        y = next_batch.get("reward") + (1 - next_batch.get("done").float()) * gamma * target_q
    
    q1_val = q1(batch).get("q_value")
    q2_val = q2(batch).get("q_value")
    q_loss = nn.functional.mse_loss(q1_val, y) + nn.functional.mse_loss(q2_val, y)

    #with torch.no_grad():
    #    td_error = torch.min(
    #        (q1_val - y).abs(),
    #        (q2_val - y).abs(),
    #    ).squeeze(-1)

    #buffer.update_priority(info["index"], td_error)

    Q_optimizers[i].zero_grad()
    q_loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 1.0)
    Q_optimizers[i].step()
    
      
    actor_td = actor(batch)
    log_prob = torch.clamp(actor_td.get("sample_log_prob"), -7, 100)
    
    q1_pi = q1(actor_td).get("q_value")
    q2_pi = q2(actor_td).get("q_value")
    min_q_pi = torch.min(q1_pi, q2_pi)

    #gate_value = actor_td.get("action")[:, 4:5]

    # MASKED ENTROPY: If gate is 0, we don't care about entropy as much
    # We add a small epsilon (0.1) so it doesn't go to zero entirely
    #entropy_mask = (1-gate_value.detach())
    
    policy_loss = (alpha.detach() * log_prob - min_q_pi).mean()

    policy_optimizers[i].zero_grad()
    policy_loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    policy_optimizers[i].step()
    
    #reconstruction = decoder_net(r_feat)
    #recon_loss = nn.functional.mse_loss(reconstruction, batch.get("observation"))

    #optimizer_decoder.zero_grad()
    #recon_loss.backward()

    #torch.nn.utils.clip_grad_norm_(decoder_net.parameters(), 1.0)
    #optimizer_decoder.step()

    # Alpha Update
    
    alpha_loss = -(log_alpha * (log_prob + target_entropy[i]).detach()).mean()
    alpha_optimizers[i].zero_grad()
    alpha_loss.backward()
    alpha_optimizers[i].step()

    #with torch.no_grad():
    #    log_alpha.clamp_(-9.0, -1.0)
    

    # Soft Update
    for t1, t2, s1, s2 in zip(t_q1.parameters(), t_q2.parameters(), q1.parameters(), q2.parameters()):
        t1.data.copy_(t1.data * (1.0 - tau) + s1.data * tau)
        t2.data.copy_(t2.data * (1.0 - tau) + s2.data * tau)
        
    return q_loss.item(), policy_loss.item(), log_alpha.item()

def save_checkpoint(path):
    checkpoint = {
        'policies': [p.state_dict() for p in policy_modules],
        'critics': [q.state_dict() for q in Q_modules],
        'log_alphas': [la for la in log_alphas],
        'opt_pols': [opt.state_dict() for opt in policy_optimizers],
        'opt_qs': [opt.state_dict() for opt in Q_optimizers],
        'opt_alphas': [opt.state_dict() for opt in alpha_optimizers],
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(path):
    checkpoint = torch.load(path)
    
    for i in range(2):
        policy_modules[i].load_state_dict(checkpoint['policies'][i])
        log_alphas[i].data.copy_(checkpoint['log_alphas'][i].data)
        policy_optimizers[i].load_state_dict(checkpoint['opt_pols'][i])
        alpha_optimizers[i].load_state_dict(checkpoint['opt_alphas'][i])
        
    for i in range(4):
        Q_modules[i].load_state_dict(checkpoint['critics'][i])
        # Targets should match the loaded Q nets initially
        Q_modules_target[i].load_state_dict(checkpoint['critics'][i])
        
    for i in range(2):
         Q_optimizers[i].load_state_dict(checkpoint['opt_qs'][i])
            
    for i in range(2):
        for param_group in policy_optimizers[i].param_groups:
            param_group['lr'] = policy_lrs[i]
    
    for i in range(2):
        for param_group in Q_optimizers[i].param_groups:
            param_group['lr'] = Q_lrs[i]

    for i in range(2):
        for param_group in alpha_optimizers[i].param_groups:
            param_group['lr'] = 1e-5
    
    print(f"Loaded checkpoint from {path}")

if load_filepath is not None:
    load_checkpoint(load_filepath)

envs = 2048 #2048
if not train:
    envs = 3

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
    noise_seeds[:,i,0] = (np.random.rand(envs) * (2000-240)).astype(np.int16)
    noise_seeds[:,i,1] = (np.random.rand(envs) * (1000-120)).astype(np.int16)

puck_noise = np.empty((envs, 2))
for i in range(2):
    noise_idx = (puck_pos*100).astype(np.int16) + noise_seeds[:,i,:]
    noise_idx[:,0] = np.clip(noise_idx[:,0], 0, 1999)
    noise_idx[:,1] = np.clip(noise_idx[:,1], 0, 999)
    puck_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]

mallet_noise = np.empty((envs, 4)) #mallet, op mallet
for i in range(4):
    noise_idx = (mallet_init[int(i/2)]*100).astype(np.int16) + noise_seeds[:,2+i%2,:]
    noise_idx[:,0] = np.clip(noise_idx[:,0], 0, 1999)
    noise_idx[:,1] = np.clip(noise_idx[:,1], 0, 999)
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

ab_vars[:,:,3:] = ab_vars[:,:,3:] / np.abs(ab_vars[:,:,3:]) * np.minimum(np.abs(ab_vars[:,:,3:]), np.abs(ab_vars[:,:,:3])*9.5/10)

ab_vars *= np.random.uniform(speed_var[0], speed_var[1], size=(envs, 2, 1))

if not train:
    ab_vars[1,0,:] = np.array([2.259685e-05, 1.0497799e-02, 6.2993579e-02, -1.256968e-05, -5.087524e-03, 4.398169e-04])
    ab_vars[1,1,:] = np.array([2.259685e-05, 1.0497799e-02, 6.2993579e-02, -1.256968e-05, -5.087524e-03, 4.398169e-04])
    ab_vars[0,0,:] =np.array([2.259685e-05, 1.0497799e-02, 6.2993579e-02, -1.256968e-05, -5.087524e-03, 4.398169e-04])
    ab_vars[0,1,:] = np.array([2.259685e-05, 1.0497799e-02, 6.2993579e-02, -1.256968e-05, -5.087524e-03, 4.398169e-04])

ab_obs = np.zeros((2*envs, 6))

ab_obs_scaling = np.array([
        0.42*1e4,   # a1: ~1e-6 * 1e4 = 1e-2
        1e1,   # a2: ~1e-3 * 1e1 = 1e-2 (keep as is since it's already reasonable)
        1e0,   # a3: ~1e-2 * 1e0 = 1e-2 (keep as is)
        0.73*1e4,   # b1: ~1e-6 * 1e4 = 1e-2
        1e1,   # b2: ~1e-3 * 1e1 = 1e-2 (keep as is)
        0.8*1e1    # b3: ~1e-3 * 1e1 = 1e-2
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

camera_buffer_size = 50

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

beam_interference = np.empty((2*envs), dtype=np.bool_)
large_beam_interference = np.empty((2*envs), dtype=np.bool_)
beam_interference[:envs] = np.logical_and(puck_pos[:,0]+puck_r > beam_coeffs[0,0] * np.full((envs,2), mallet_init[0])[:,0] + beam_coeffs[0,1], puck_pos[:,0]-puck_r < beam_coeffs[1,0] * np.full((envs,2), mallet_init[0])[:,0] + beam_coeffs[1,1])
beam_interference[envs:] = np.logical_and(bounds[0]-puck_pos[:,0]+puck_r > beam_coeffs[0,0] * np.full((envs,2), bounds[0]-mallet_init[1])[:,0] + beam_coeffs[0,1], bounds[0]-puck_pos[:,0]-puck_r < beam_coeffs[1,0] * np.full((envs,2), bounds[0]-mallet_init[1])[:,0] + beam_coeffs[1,1])

camera_buffer[:, :envs, :2] = np.tile(puck_pos+puck_noise,(camera_buffer_size,1,1)) +\
                                     np.stack((np.random.normal(0,\
                                                np.where(beam_interference[:envs], puck_beam_std[0,0] * puck_pos[:,0] + puck_beam_std[0,1], puck_std[0,0] * puck_pos[:,0] + puck_std[0,1]),\
                                                 (camera_buffer_size,envs)),\
                                             np.random.normal(0,\
                                                np.where(beam_interference[:envs], puck_beam_std[1,0] * puck_pos[:,0] + puck_beam_std[1,1], puck_std[1,0] * puck_pos[:,0] + puck_std[1,1]),\
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
                                np.random.rand(envs, 1)*12+4,
                                np.random.rand(envs, 1)*12+4,
                                ab_obs[:envs, :],
                                np.ones((envs,1))], axis=1)

#puck_std = [[0,0.0005], [0,0.0005]] # std in x y, a*x + b

obs[envs:, :] = np.concatenate([np.zeros((envs,20)),
                                np.full((envs,2), mallet_init[0]),
                                np.zeros((envs,2)),
                                np.full((envs,2), mallet_init[0]),
                                np.zeros((envs,2)),
                                np.full((envs,2), mallet_init[0]),
                                np.random.rand(envs, 1)*12+4,
                                np.random.rand(envs, 1)*12+4,
                                ab_obs[envs:, :],
                                np.zeros((envs,1))], axis=1)

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
miss_mask = np.empty((2*envs))

if train:
    self_play = 512
else:
    self_play = 1

dones = np.zeros((2*envs,), dtype=np.bool_)
err_dones = np.zeros((envs,), dtype=np.bool_)
rewards = np.zeros((2*envs,))
terminal_rewards = np.zeros((2*envs,))
close_side = np.ones((envs,), dtype=np.bool_)
defending = np.zeros((2*envs,), dtype=np.bool_)
defending[envs+self_play:] = True
def_init = defending.copy()[:,None]
defending_mask = np.zeros((2*envs,), dtype=np.bool_)
defending_mask[envs+self_play:] = True
next_state_def = defending.copy()
attack_mask = np.ones((2*envs), dtype=np.bool_)

crossed = np.zeros((2*envs,))

past_xf = np.stack([obs[:envs, 28:30], obs[envs:, 28:30]], axis=1)
past_xf[:,1,:2] = bounds-past_xf[:,1,:2]
past_Vo = np.stack([obs[:envs, 30:32], obs[envs:, 30:32]], axis=1)

xf_mallet_noise = np.empty((envs, 4))
actions_since_reset = 0

flipped = np.random.random((2*envs,)) > 0.5

random_offset_attack = (np.random.uniform(size=(2*envs,))-0.5)*0#0.06
random_offset_defending = (np.random.uniform(size=(2*envs,))-0.5)*0#0.03
random_offset = (np.random.uniform(size=(2*envs,))-0.5)*0#0.03
random_offset[:] = random_offset_defending
past_puck_pos = puck_pos.copy()

action_skips = np.zeros((envs-self_play,), dtype=np.int8)

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
        mallet_pos, _, puck_pos, puck_vel, _, _ = sim.step(next_img)
        for i in range(2):
            noise_idx = (puck_pos*100).astype(np.int16) + noise_seeds[:,i,:]
            noise_idx[:,0] = np.clip(noise_idx[:,0], 0, 1999)
            noise_idx[:,1] = np.clip(noise_idx[:,1], 0, 999)
            puck_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]

        beam_interference[:envs] = np.logical_and(puck_pos[:,0]+puck_r > beam_coeffs[0,0] * mallet_pos[:,0,0] + beam_coeffs[0,1], puck_pos[:,0]-puck_r < beam_coeffs[1,0] * mallet_pos[:,0,0] + beam_coeffs[1,1])

        beam_interference[envs:] = np.logical_and(bounds[0]-puck_pos[:,0]+puck_r > beam_coeffs[0,0] * (bounds[0]-mallet_pos[:,1,0]) + beam_coeffs[0,1], bounds[0]-puck_pos[:,0]-puck_r < beam_coeffs[1,0] * (bounds[0]-mallet_pos[:,1,0]) + beam_coeffs[1,1])

        puck_wgn_left = np.stack((np.random.normal(0, np.where(beam_interference[:envs], puck_beam_std[0,0] * puck_pos[:,0] + puck_beam_std[0,1], puck_std[0,0] * puck_pos[:,0] + puck_std[0,1]), (envs,)), np.random.normal(0, np.where(beam_interference[:envs], puck_beam_std[1,0] * puck_pos[:,0] + puck_beam_std[1,1], puck_std[1,0] * puck_pos[:,0] + puck_std[1,1]), (envs,))),axis=1)

        puck_wgn_right = np.stack((np.random.normal(0, np.where(beam_interference[envs:], puck_beam_std[0,0] * (bounds[0]-puck_pos[:,0]) + puck_beam_std[0,1], puck_std[0,0] * (bounds[0]-puck_pos[:,0]) + puck_std[0,1]), (envs,)),np.random.normal(0, np.where(beam_interference[envs:], puck_beam_std[1,0] * (bounds[0]-puck_pos[:,0]) + puck_beam_std[1,1], puck_std[1,0] * (bounds[0]-puck_pos[:,0]) + puck_std[1,1]), (envs,))),axis=1)

        large_beam_interference[:] = np.concatenate((np.logical_and(puck_pos[:,0] > (beam_coeffs[0,0] * mallet_pos[:,0,0] + beam_coeffs[0,1]), puck_pos[:,0] < (beam_coeffs[1,0] * mallet_pos[:,0,0] + beam_coeffs[1,1])),
            np.logical_and((bounds[0]-puck_pos[:,0]) > (beam_coeffs[0,0] * (bounds[0]-mallet_pos[:,1,0]) + beam_coeffs[0,1]), (bounds[0]-puck_pos[:,0]) < (beam_coeffs[1,0] * (bounds[0]-mallet_pos[:,1,0]) + beam_coeffs[1,1]))), axis=0)

        beam_interference[envs+self_play:] = np.logical_not(defending[envs+self_play:]) & beam_interference[envs+self_play:]
        large_beam_interference[envs+self_play:] = np.logical_not(defending[envs+self_play:]) & large_beam_interference[envs+self_play:]

        if large_beam_interference.any():
            past_puck_data = camera_buffer.get([0])[:,:2] # player, x/y (envs,2)
            miss_mask[:envs] = np.logical_or(np.random.random((envs)) < percent_miss[0] * puck_pos[:,0] + percent_miss[1],\
                                       np.logical_and(puck_pos[:,0] - puck_r > beam_coeffs[0,0] * mallet_pos[:,0,0] + beam_coeffs[0,1], puck_pos[:,0] + puck_r < beam_coeffs[1,0] * mallet_pos[:,0,0] + beam_coeffs[1,1]))
            miss_mask[envs:] = np.logical_or(np.random.random((envs)) < percent_miss[0] * (bounds[0] - puck_pos[:,0]) + percent_miss[1],\
                                       np.logical_and(bounds[0] - puck_pos[:,0] - puck_r > beam_coeffs[0,0] * (bounds[0] - mallet_pos[:,1,0]) + beam_coeffs[0,1], bounds[0] - puck_pos[:,0] + puck_r < beam_coeffs[1,0] * (bounds[0] - mallet_pos[:,1,0]) + beam_coeffs[1,1]))

            puck_pos = np.concatenate([puck_pos + puck_noise + puck_wgn_left, bounds - puck_pos - puck_noise - puck_wgn_right],axis=0)
            puck_pos[np.logical_and(large_beam_interference, miss_mask)] = past_puck_data[np.logical_and(large_beam_interference, miss_mask)]
        else:
            puck_pos = np.concatenate([puck_pos + puck_noise + puck_wgn_left, bounds - puck_pos - puck_noise - puck_wgn_right],axis=0)

        #mallet_noise = np.empty((envs, 4)) #mallet, op mallet
        for i in range(2,4):
            noise_idx = (mallet_pos[:,int(i/2),:]*100).astype(np.int16) + noise_seeds[:,2+i%2,:]
            noise_idx[:,0] = np.clip(noise_idx[:,0], 0, 1999)
            noise_idx[:,1] = np.clip(noise_idx[:,1], 0, 999)
            mallet_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]
        mallet_wgn_noise = np.stack((np.random.normal(0, puck_std[0,0] * mallet_pos[:,1,0] + puck_std[0,1], (envs,)), np.random.normal(0, puck_std[1,0] * mallet_pos[:,1,0] + puck_std[1,1], (envs,))), axis=-1)

        op_mallet_pos = np.concatenate([mallet_pos[:,1,:] + mallet_noise[:,2:] + mallet_wgn_noise, bounds - mallet_pos[:,0,:] - mallet_noise[:,:2]], axis=0)
        camera_buffer.put(np.concatenate([puck_pos, op_mallet_pos], axis=1))

        time_from_last_img = 0
        agent_actions.subtract(next_img)
        mallet_time.subtract(next_img)
        agent_actions.put(np.clip(np.random.normal(image_delay[0], image_delay[1]), image_delay[2], image_delay[3]))
        mallet_time.put(agent_actions.get(0) - np.clip(np.random.normal(mallet_delay[0], mallet_delay[1]), mallet_delay[2], mallet_delay[3]))

        inference_img.put(np.logical_not(inference_img.get(0)))
    elif next_mallet < next_img and next_mallet < next_action:
        mallet_pos, mallet_vel, _, _, _, _ = sim.step(next_mallet)

        #(2*envs, 2)
        past_obs[:,20:22] = np.concatenate([mallet_pos[:,0,:] + np.random.normal(0, mallet_std, (envs,2)), bounds - mallet_pos[:,1,:] + np.random.normal(0, mallet_std, (envs,2))], axis=0)
        vel_noise = np.random.normal(0, mallet_vel_std, size=(envs,2,2))
        past_obs[:,22:24] = np.concatenate([mallet_vel[:,0,:] + vel_noise[:,0,:], -mallet_vel[:,1,:] + vel_noise[:,1,:]], axis=0)
        
        time_from_last_img -= next_mallet
        agent_actions.subtract(next_mallet)
        mallet_time.subtract(next_mallet)

    elif next_action < next_img and next_action < next_mallet:
        _, _, puck_pos, _, _, _ = sim.step(next_action)

        #[puck_pos, opp_mallet_pos] #guess mean low std
        #[mallet_pos, mallet_vel, past mallet_pos, past_mallet_vel, past action (x0, V)]
        #[e_n0, e_nr, e_nf, e_t0, e_tr, e_tf, var] x2 mallet and wall,

        # (2*envs, 20)
        camera_obs = camera_buffer.get(indices=[img_idx, img_idx+1, img_idx+2, img_idx+5, img_idx+11])[:,:]
        past_obs[:,:20] = camera_obs

        time_from_last_img -= next_action
        agent_actions.subtract(next_action)
        mallet_time.subtract(next_action)

        break

if train:

    """
    policy_buffer = PrioritizedReplayBuffer(
        storage=LazyTensorStorage(1_500_000),
        alpha=0.2,        # how strong prioritization is
        beta=0.4,         # importance-sampling correction
        batch_size=batch_size,
    )

    def_buffer = PrioritizedReplayBuffer(
        storage=LazyTensorStorage(1_500_000),
        alpha=0.2,        # how strong prioritization is
        beta=0.4,         # importance-sampling correction
        batch_size=batch_size,
    )
    """

    policy_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(1_500_000),
        batch_size=batch_size,
    )

    def_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(1_500_000),
        batch_size=batch_size,
    )

    #Simulation loop
    for update in range(500000):  # Training loop
        print(update)
        print("simulating...")

        for timestep in range(10):
            random_offset[defending] = random_offset_defending[defending]
            random_offset[np.logical_not(defending)] = random_offset_attack[np.logical_not(defending)]
            with torch.no_grad():

                past_obs[:, 21] += random_offset
                past_obs[:, 25] += random_offset

                flipped = np.random.random((2*envs,)) > 0.5
                past_obs[flipped] = apply_symmetry(past_obs[flipped])

                tensor_obs = TensorDict({"observation": torch.tensor(past_obs, dtype=torch.float32)}).to('cuda')

                policy_out = policy_modules[0](tensor_obs)
                actions = policy_out["action"].detach().to('cpu')

                actions_np = actions.numpy()    

                def_tensor_obs = TensorDict({"observation": torch.tensor(past_obs[envs+self_play:], dtype=torch.float32)}).to('cuda')
                def_out = policy_modules[1](def_tensor_obs)
                def_actions = def_out["action"].detach().to('cpu')

                actions_np[defending & defending_mask] = def_actions[defending[envs+self_play:]]

                past_obs[flipped] = apply_symmetry(past_obs[flipped])
                actions_np[:,1][flipped] = bounds[1] - actions_np[:,1][flipped]

                past_obs[:, 21] -= random_offset
                past_obs[:, 25] -= random_offset
                    
            xf = np.stack([actions_np[:envs, :2], actions_np[envs:,:2]], axis=1)
            

            xf[:,0,0] = np.where((xf[:,0,0] < (mallet_r + 2*puck_r + 0.01)) & (past_obs[:envs,0] < obs[:envs, 20]), mallet_r + 2*puck_r + 0.01, xf[:,0,0])
            xf[:,0,1] = np.where((xf[:,0,1] < (mallet_r + 2*puck_r + 0.01)) & (past_obs[:envs,1] < obs[:envs, 21]), mallet_r + 2*puck_r + 0.01, xf[:,0,1])
            xf[:,0,1] = np.where((xf[:,0,1] > (bounds[1] - mallet_r - 2*puck_r - 0.01)) & (past_obs[:envs,1] > obs[:envs, 21]), bounds[1] - mallet_r - 2*puck_r - 0.01, xf[:,0,1])

            xf[:,1,0] = np.where((xf[:,1,0] < (mallet_r + 2*puck_r + 0.01)) & (past_obs[envs:,0] < obs[envs:, 20]), mallet_r + 2*puck_r + 0.01, xf[:,1,0])
            xf[:,1,1] = np.where((xf[:,1,1] < (mallet_r + 2*puck_r + 0.01)) & (past_obs[envs:,1] < obs[envs:, 21]), mallet_r + 2*puck_r + 0.01, xf[:,1,1])
            xf[:,1,1] = np.where((xf[:,1,1] > (bounds[1] - mallet_r - 2*puck_r - 0.01)) & (past_obs[envs:,1] > obs[envs:, 21]), bounds[1] - mallet_r - 2*puck_r - 0.01, xf[:,1,1])

            xf[:,1,:] = bounds - xf[:,1,:]

            Vo = actions_np[:,2][:, None] * Vmax * np.stack((1+actions_np[:,3], 1-actions_np[:,3]), axis=1) + np.stack((np.random.normal(0, V_std_x, (2*envs)), np.random.normal(0, V_std_y, (2*envs))), axis=-1)

            Vo[:,0] = np.maximum(Vo[:,0], 0.1)
            Vo[:,1] = np.maximum(Vo[:,1], 0.1)
            Vo = np.stack([Vo[:envs,:], Vo[envs:,:]], axis=1)

            rewards[:] += 0.3 * actions_np[:,4]
            no_update_mask = np.stack([actions_np[:envs,4] > np.random.random((envs,))*1.01, np.concatenate([actions_np[envs:envs+self_play,4] > np.random.random((self_play,))*1.01, actions_np[envs+self_play:,4] > np.random.random((envs-self_play,))*1.01], axis=0)], axis=1)
            action_skips += (action_skips != 0)
            action_skips += (action_skips == 0) & (actions_np[envs+self_play:,4] > 0.5)
            action_skips[action_skips == 5] = 0
            action_skips[np.logical_not(defending[envs+self_play:])] = 0

            no_update_mask[self_play:,1][action_skips != 0] = True

            xf[no_update_mask] = past_xf[no_update_mask]
            Vo[no_update_mask] = past_Vo[no_update_mask]

            past_xf = xf.copy()
            past_Vo = Vo.copy()

            sim.take_action(xf, Vo)
            sim.impulse(2/60, 0.005)

            obs[:, 24:28] = past_obs[:, 20:24]
            obs[:, 28:32] = np.concatenate([np.concatenate([xf[:,0,:], Vo[:,0,:]], axis=1), np.concatenate([bounds - xf[:,1,:], Vo[:,1,:]], axis=1)], axis=0)

            attack_mask[envs+self_play:] = np.logical_not(defending[envs+self_play:])

            rewards[attack_mask] += np.where((actions_np[attack_mask,0] < mallet_r+0.02) | (actions_np[attack_mask,0] > bounds[0]/2-mallet_r-0.02), -1.0, 0)
            rewards[attack_mask] += np.where((actions_np[attack_mask,1] < mallet_r+0.02) | (actions_np[attack_mask,1] > bounds[1] - mallet_r - 0.02), -1.0, 0)

            rewards[attack_mask] += np.where((actions_np[attack_mask,2] > 0.995) | (actions_np[attack_mask,2] < 0.005), -1.0, 0)
            rewards[attack_mask] += np.where((actions_np[attack_mask,3] > 0.995) | (actions_np[attack_mask,3] < -0.995), -1.0, 0)

            rewards[:] += np.where((actions_np[:,4] > 0.995) | (actions_np[:,4] < 0.005), -1.0, 0)

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
                    mallet_pos, _, puck_pos, puck_vel, cL, cR = sim.step(next_img, calculate_cross_left=True)

                    crossed = np.maximum(crossed, np.concatenate((cR, cL), axis=0))

                    env_err = sim.check_state()
                    entered_left_goal_mask, entered_right_goal_mask = sim.check_goal()
                    if len(env_err) > 0:
                        print("err")
                        print(env_err)
                        for idx in env_err:
                            err_dones[idx] = True
                            if puck_pos[idx, 0] < bounds[0]/2:
                                terminal_rewards[idx] -= 300.0
                            elif idx < self_play:
                                terminal_rewards[envs+idx] -= 300.0

                    terminal_rewards[:envs][entered_left_goal_mask] -= 300
                    terminal_rewards[:envs][entered_right_goal_mask] += 400

                    terminal_rewards[envs:envs+self_play][entered_left_goal_mask[:self_play]] += 400
                    terminal_rewards[envs:][entered_right_goal_mask] -= 300.0

                    err_dones[entered_left_goal_mask | entered_right_goal_mask] = True

                    new_dones_fail = (np.logical_not(dones[envs+self_play:])) & (defending[envs+self_play:] & (puck_pos[self_play:,0] < bounds[0]/2) & (past_puck_pos[self_play:,0] > bounds[0]/2))
                    new_dones_success = (np.logical_not(dones[envs+self_play:])) & (defending[envs+self_play:] & (puck_pos[self_play:,0] > bounds[0]/2) & (np.linalg.norm(puck_vel[self_play:], axis=1) < 0.5))
                    dones[envs+self_play:] = dones[envs+self_play:] | new_dones_fail | new_dones_success
                    terminal_rewards[envs+self_play:][new_dones_fail] += -10
                    terminal_rewards[envs+self_play:][new_dones_success] += 100
        
                    for i in range(2):
                        noise_idx = (puck_pos*100).astype(np.int16) + noise_seeds[:,i,:]
                        noise_idx[:,0] = np.clip(noise_idx[:,0], 0, 1999)
                        noise_idx[:,1] = np.clip(noise_idx[:,1], 0, 999)
                        puck_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]

                    beam_interference[envs:] = np.logical_and(bounds[0]-puck_pos[:,0]+puck_r > beam_coeffs[0,0] * (bounds[0]-mallet_pos[:,1,0]) + beam_coeffs[0,1], bounds[0]-puck_pos[:,0]-puck_r < beam_coeffs[1,0] * (bounds[0]-mallet_pos[:,1,0]) + beam_coeffs[1,1])

                    puck_wgn_left = np.stack((np.random.normal(0, np.abs(np.where(beam_interference[:envs], puck_beam_std[0,0] * puck_pos[:,0] + puck_beam_std[0,1], puck_std[0,0] * puck_pos[:,0] + puck_std[0,1])), (envs,)), np.random.normal(0, np.abs(np.where(beam_interference[:envs], puck_beam_std[1,0] * puck_pos[:,0] + puck_beam_std[1,1], puck_std[1,0] * puck_pos[:,0] + puck_std[1,1])), (envs,))),axis=1)

                    puck_wgn_right = np.stack((np.random.normal(0, np.abs(np.where(beam_interference[envs:], puck_beam_std[0,0] * (bounds[0]-puck_pos[:,0]) + puck_beam_std[0,1], puck_std[0,0] * (bounds[0]-puck_pos[:,0]) + puck_std[0,1])), (envs,)),np.random.normal(0, np.abs(np.where(beam_interference[envs:], puck_beam_std[1,0] * (bounds[0]-puck_pos[:,0]) + puck_beam_std[1,1], puck_std[1,0] * (bounds[0]-puck_pos[:,0]) + puck_std[1,1])), (envs,))),axis=1)

                    large_beam_interference[:] = np.concatenate((np.logical_and(puck_pos[:,0] > (beam_coeffs[0,0] * mallet_pos[:,0,0] + beam_coeffs[0,1]), puck_pos[:,0] < (beam_coeffs[1,0] * mallet_pos[:,0,0] + beam_coeffs[1,1])),
                        np.logical_and((bounds[0]-puck_pos[:,0]) > (beam_coeffs[0,0] * (bounds[0]-mallet_pos[:,1,0]) + beam_coeffs[0,1]), (bounds[0]-puck_pos[:,0]) < (beam_coeffs[1,0] * (bounds[0]-mallet_pos[:,1,0]) + beam_coeffs[1,1]))), axis=0)

                    beam_interference[envs+self_play:] = np.logical_not(defending[envs+self_play:]) & beam_interference[envs+self_play:]
                    large_beam_interference[envs+self_play:] = np.logical_not(defending[envs+self_play:]) & large_beam_interference[envs+self_play:]

                    if large_beam_interference.any():
                        past_puck_data = camera_buffer.get([0])[:,:2] # player, x/y (envs,2)
                        miss_mask[:envs] = np.logical_or(np.random.random((envs)) < percent_miss[0] * puck_pos[:,0] + percent_miss[1],\
                                                   np.logical_and(puck_pos[:,0] - puck_r > beam_coeffs[0,0] * mallet_pos[:,0,0] + beam_coeffs[0,1], puck_pos[:,0] + puck_r < beam_coeffs[1,0] * mallet_pos[:,0,0] + beam_coeffs[1,1]))
                        miss_mask[envs:] = np.logical_or(np.random.random((envs)) < percent_miss[0] * (bounds[0] - puck_pos[:,0]) + percent_miss[1],\
                                                   np.logical_and(bounds[0] - puck_pos[:,0] - puck_r > beam_coeffs[0,0] * (bounds[0] - mallet_pos[:,1,0]) + beam_coeffs[0,1], bounds[0] - puck_pos[:,0] + puck_r < beam_coeffs[1,0] * (bounds[0] - mallet_pos[:,1,0]) + beam_coeffs[1,1]))

                        puck_pos = np.concatenate([puck_pos + puck_noise + puck_wgn_left, bounds - puck_pos - puck_noise - puck_wgn_right],axis=0)
                        puck_pos[np.logical_and(large_beam_interference, miss_mask)] = past_puck_data[np.logical_and(large_beam_interference, miss_mask)]
                    else:
                        puck_pos = np.concatenate([puck_pos + puck_noise + puck_wgn_left, bounds - puck_pos - puck_noise - puck_wgn_right],axis=0)

                    #mallet_noise = np.empty((envs, 4)) #mallet, op mallet
                    for i in range(2,4):
                        noise_idx = (mallet_pos[:,int(i/2),:]*100).astype(np.int16) + noise_seeds[:,2+i%2,:]
                        noise_idx[:,0] = np.clip(noise_idx[:,0], 0, 1999)
                        noise_idx[:,1] = np.clip(noise_idx[:,1], 0, 999)
                        mallet_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]
                    mallet_wgn_noise = np.stack((np.random.normal(0, np.abs(puck_std[0,0] * mallet_pos[:,1,0] + puck_std[0,1]), (envs,)), np.random.normal(0, np.abs(puck_std[1,0] * mallet_pos[:,1,0] + puck_std[1,1]), (envs,))), axis=-1)

                    op_mallet_pos = np.concatenate([mallet_pos[:,1,:] + mallet_noise[:,2:] + mallet_wgn_noise, bounds - mallet_pos[:,0,:] - mallet_noise[:,:2]], axis=0)

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
                    mallet_pos, mallet_vel, puck_pos, puck_vel, cL, cR = sim.step(next_mallet, calculate_cross_left=True)

                    crossed = np.maximum(crossed, np.concatenate((cR, cL), axis=0))

                    env_err = sim.check_state()
                    entered_left_goal_mask, entered_right_goal_mask = sim.check_goal()
                    if len(env_err) > 0:
                        print("err")
                        print(env_err)
                        for idx in env_err:
                            err_dones[idx] = True
                            if puck_pos[idx, 0] < bounds[0]/2:
                                terminal_rewards[idx] -= 300.0
                            elif idx < self_play:
                                terminal_rewards[envs+idx] -= 300.0

                    terminal_rewards[:envs][entered_left_goal_mask] -= 300
                    terminal_rewards[:envs][entered_right_goal_mask] += 400

                    terminal_rewards[envs:envs+self_play][entered_left_goal_mask[:self_play]] += 400
                    terminal_rewards[envs:][entered_right_goal_mask] -= 300

                    err_dones[entered_left_goal_mask | entered_right_goal_mask] = True

                    new_dones_fail = (np.logical_not(dones[envs+self_play:])) & (defending[envs+self_play:] & (puck_pos[self_play:,0] < bounds[0]/2) & (past_puck_pos[self_play:,0] > bounds[0]/2))
                    new_dones_success = (np.logical_not(dones[envs+self_play:])) & (defending[envs+self_play:] & (puck_pos[self_play:,0] > bounds[0]/2) & (np.linalg.norm(puck_vel[self_play:], axis=1) < 0.5))
                    dones[envs+self_play:] = dones[envs+self_play:] | new_dones_fail | new_dones_success
                    terminal_rewards[envs+self_play:][new_dones_fail] += -10
                    terminal_rewards[envs+self_play:][new_dones_success] += 100

                    mal_noise = np.random.normal(0, mallet_std, (envs,2,2))
                    obs[:,20:22] = np.concatenate([mallet_pos[:,0,:] + mal_noise[:,0,:], bounds - mallet_pos[:,1,:] + mal_noise[:,1,:]], axis=0)
                    vel_noise = np.random.normal(0, mallet_vel_std, size=(envs,2,2))
                    obs[:,22:24] = np.concatenate([mallet_vel[:,0,:] + vel_noise[:,0,:], -mallet_vel[:,1,:] + vel_noise[:,1,:]], axis=0)
                    
                    time_from_last_img -= next_mallet
                    agent_actions.subtract(next_mallet)
                    mallet_time.subtract(next_mallet)

                elif next_action < next_img and next_action < next_mallet:
                    mallet_pos, _, puck_pos, puck_vel, cL, cR = sim.step(next_action, calculate_cross_left=True)

                    crossed = np.maximum(crossed, np.concatenate((cR, cL), axis=0))

                    env_err = sim.check_state()
                    entered_left_goal_mask, entered_right_goal_mask = sim.check_goal()
                    if len(env_err) > 0:
                        print("err")
                        print(env_err)
                        for idx in env_err:
                            err_dones[idx] = True
                            if puck_pos[idx, 0] < bounds[0]/2:
                                terminal_rewards[idx] -= 300.0
                            elif idx < self_play:
                                terminal_rewards[envs+idx] -= 300.0

                    terminal_rewards[:envs][entered_left_goal_mask] -= 300
                    terminal_rewards[:envs][entered_right_goal_mask] += 400

                    terminal_rewards[envs:envs+self_play][entered_left_goal_mask[:self_play]] += 400
                    terminal_rewards[envs:][entered_right_goal_mask] -= 300
                    
                    err_dones[entered_left_goal_mask | entered_right_goal_mask] = True

                    new_dones_fail = (np.logical_not(dones[envs+self_play:])) & (defending[envs+self_play:] & (puck_pos[self_play:,0] < bounds[0]/2) & (past_puck_pos[self_play:,0] > bounds[0]/2))
                    new_dones_success = (np.logical_not(dones[envs+self_play:])) & (defending[envs+self_play:] & (puck_pos[self_play:,0] > bounds[0]/2) & (np.linalg.norm(puck_vel[self_play:], axis=1) < 0.5))
                    dones[envs+self_play:] = dones[envs+self_play:] | new_dones_fail | new_dones_success
                    terminal_rewards[envs+self_play:][new_dones_fail] += -10
                    terminal_rewards[envs+self_play:][new_dones_success] += 100

                    camera_obs = camera_buffer.get(indices=[img_idx, img_idx+1, img_idx+2, img_idx+5, img_idx+11])
                    obs[:,:20] = camera_obs

                    next_state_def[envs:] = (puck_pos[:,0] < bounds[0]/2) | (defending[envs:] & (np.linalg.norm(puck_vel, axis=1) > 0.5))
                    next_state_def[:envs] = (puck_pos[:,0] > bounds[0]/2) | (defending[:envs] & (np.linalg.norm(puck_vel, axis=1) > 0.5))
                    camera_obs = camera_buffer.get(indices=[img_idx+img_offset, img_idx+img_offset+1, img_idx+img_offset+2, img_idx+img_offset+5, img_idx+img_offset+11])[envs+self_play:][next_state_def[envs+self_play:]]
                    obs[envs+self_play:,:20][next_state_def[envs+self_play:]] = camera_obs

                    time_from_last_img -= next_action
                    agent_actions.subtract(next_action)
                    mallet_time.subtract(next_action)

                    break

            actions_since_reset += 1
            attack_mask[envs+self_play:] = np.logical_not(defending[envs+self_play:]) & np.logical_not(next_state_def[envs+self_play:])

            rewards[:envs] += np.where((puck_pos[:,0] < bounds[0]/2) & (np.abs(puck_vel[:,0]) < 0.15), -2.2, 0)
            rewards[envs:] += np.where((puck_pos[:,0] > bounds[0]/2) & (np.abs(puck_vel[:,0]) < 0.15), -2.2, 0)

            rewards[:envs][np.logical_not(defending[:envs]) & (puck_pos[:,0] > bounds[0]/2)] += (np.linalg.norm(puck_vel,axis=1)*8.5 * crossed[:envs])[np.logical_not(defending[:envs]) & (puck_pos[:,0] > bounds[0]/2)]
            rewards[envs:][np.logical_not(defending[envs:]) & (puck_pos[:,0] < bounds[0]/2)] += (np.linalg.norm(puck_vel,axis=1)*8.5 * crossed[envs:])[np.logical_not(defending[envs:]) & (puck_pos[:,0] < bounds[0]/2)]

            rewards[:envs] += np.where((puck_pos[:,0] < bounds[0]/2), -0.7, 0)
            rewards[envs:] += np.where((puck_pos[:,0] > bounds[0]/2) & attack_mask[envs:], -0.7, 0)

            rewards[:envs][(past_puck_pos[:,0] < bounds[0]/2) & (puck_pos[:,0] > bounds[0]/2)] -= 25.0
            rewards[envs:][(past_puck_pos[:,0] > bounds[0]/2) & (puck_pos[:,0] < bounds[0]/2)] -= 25.0

            rewards[:envs][(past_puck_pos[:,0] < bounds[0]/2) & (puck_pos[:,0] > bounds[0]/2) & (np.linalg.norm(puck_vel,axis=1) < 7.0)] -= 100.0
            rewards[envs:][attack_mask[envs:] & (past_puck_pos[:,0] > bounds[0]/2) & (puck_pos[:,0] < bounds[0]/2) & (np.linalg.norm(puck_vel,axis=1) < 5.0)] -= 100.0

            rewards[:envs][(past_puck_pos[:,0] < bounds[0]/2) & (puck_pos[:,0] > bounds[0]/2) & (np.linalg.norm(puck_vel,axis=1) < 4.0)] -= 200.0
            rewards[envs:][attack_mask[envs:] & (past_puck_pos[:,0] > bounds[0]/2) & (puck_pos[:,0] < bounds[0]/2) & (np.linalg.norm(puck_vel,axis=1) < 2.0)] -= 200.0
            #rewards[:envs][defending[:envs] & (past_puck_pos[:envs,0] < (bounds[0]/2+2*puck_r)) & (puck_pos[:envs,0] > (bounds[0]/2)+2*puck_r)] -= 10.0
            rewards[:envs+self_play][defending[:envs+self_play] & np.logical_not(next_state_def[:envs+self_play])] += 5.0
            obs[:,38] = next_state_def[:]
            dones[:envs] = err_dones #| (defending[envs:] & np.logical_not(next_state_def[envs:]))
            dones[envs:] = dones[envs:] | err_dones
            #dones[envs:envs+self_play] = dones[envs:envs+self_play] | (defending[:self_play] & np.logical_not(next_state_def[:self_play]))

            rewards[dones] += terminal_rewards[dones]
            rewards /= 25
            terminal_rewards[dones] = 0

            #print("----")
            #print(past_obs[0,:])
            #print(actions_np[0,:])
            #print(rewards[0])
            #print(obs[0,:])
            #print(dones[0])

            td = TensorDict({
                    "observation": torch.tensor(past_obs[attack_mask], dtype=torch.float32),
                    "action": torch.tensor(actions_np[attack_mask], dtype=torch.float32),
                    "next": TensorDict({
                        "reward": torch.tensor(rewards[attack_mask,None], dtype=torch.float32),
                        "observation": torch.tensor(obs[attack_mask], dtype=torch.float32),
                        "done": torch.tensor(dones[attack_mask,None], dtype=torch.bool),
                    }, batch_size=[np.sum(attack_mask)]),
                }, batch_size=[np.sum(attack_mask)])

            policy_buffer.extend(td)

            defending_td_mask = ((defending[envs+self_play:]) & (action_skips == 0)) | (defending[envs+self_play:] & np.logical_not(next_state_def[envs+self_play:]))
            
            td = TensorDict({
                    "observation": torch.tensor(past_obs[envs+self_play:][defending_td_mask], dtype=torch.float32),
                    "action": torch.tensor(actions_np[envs+self_play:][defending_td_mask], dtype=torch.float32),
                    "next": TensorDict({
                        "reward": torch.tensor(rewards[envs+self_play:, None][defending_td_mask], dtype=torch.float32),
                        "observation": torch.tensor(obs[envs+self_play:][defending_td_mask], dtype=torch.float32),
                        "done": torch.tensor(dones[envs+self_play:, None][defending_td_mask], dtype=torch.bool),
                    }, batch_size=[int(np.sum(defending_td_mask))]),
                }, batch_size=[int(np.sum(defending_td_mask))])

            def_buffer.extend(td)

            defending[:] = next_state_def
                
            sim.display_state(self_play+10)

            dones[:envs] = err_dones
            dones[envs:] = err_dones

            if actions_since_reset == 600:
                actions_since_reset = 0
                dones[:] = True

            num_resets = int(np.sum(dones[:envs]))

            if num_resets > 0:
                ab_vars = np.random.uniform(
                        low=ab_ranges[:, 0].reshape(1, 1, 6),    # Shape: (1, 1, 5)
                        high=ab_ranges[:, 1].reshape(1, 1, 6),   # Shape: (1, 1, 5)
                        size=(num_resets, 2, 6)                     # Output shape
                    )

                ab_vars[:,:,3:] = ab_vars[:,:,3:] / np.abs(ab_vars[:,:,3:]) * np.minimum(np.abs(ab_vars[:,:,3:]), np.abs(ab_vars[:,:,:3])*9.5/10)

                ab_vars *= np.random.uniform(speed_var[0], speed_var[1], size=(num_resets, 2, 1))

                ab_obs = np.zeros((2*num_resets, 6))
                    
                ab_obs[:num_resets, :] = (ab_vars[:,0,:] / pullyR) * ab_obs_scaling[:]
                ab_obs[num_resets:, :] = (ab_vars[:,1,:] / pullyR) * ab_obs_scaling[:]

                obs[dones] = np.concatenate([obs_init[dones], ab_obs, def_init[dones]], axis=1)

                camera_buffer.reset(np.where(dones)[0])

                sim.reset_sim(np.where(dones[:envs])[0], ab_vars)

                defending[envs:][dones[:envs]] = True
                defending[:envs][dones[:envs]] = False

            
            past_obs[:envs+self_play] = obs[:envs+self_play]
            past_obs[envs+self_play:][action_skips == 0] = obs[envs+self_play:][action_skips == 0]
            dones[:] = False
            rewards[:] = 0
            err_dones[:] = False
            past_puck_pos[:] = puck_pos
            crossed[:] = 0

        if update <= 40:
            continue

        print("training...")
        torch.cuda.empty_cache()

        if (len(policy_buffer) < batch_size) or (len(def_buffer) < batch_size): # or (len(def2_buffer) < batch_size):
                print("not enough samples")
                continue 
        samples = np.empty((3,int(1024*120/batch_size),3))
        for j in range(int(1024*120/batch_size)):
            #policy
            # In your training loop:

            # Accumulate gradients from all 3 agents into the encoder
            for i in range(2):
                q_loss_val, policy_loss_val, alpha_val = update_sac_agent(i, batch_size, gamma=gamma, tau=tau)
                samples[i,j,0] = q_loss_val
                samples[i,j,1] = policy_loss_val
                samples[i,j,2] = alpha_val

            if j == int(1024*120/batch_size) - 1:
                print("---")
                print(j)
                print(samples[0,:,0].mean())
                print(samples[0,:,1].mean())
                print(samples[0,:,2].mean())

        if samples[0,:,0].mean() < -100 or samples[0,:,0].mean() > 100:
            print("err")
            print(1/0)
        if update % 50 == 0:
            save_checkpoint(save_filepath)

puck_vel = np.zeros((envs,2))
puck_pos = np.zeros((envs,2))
past_puck_pos = puck_pos.copy()
random_offset = (np.random.uniform(size=(2*envs,))-0.5)*0
action_skips = np.zeros((envs-self_play,), dtype=np.int8)
#random_offset[1] = 0.05
for update in range(500000):  # Training loop
    print(update)
    print("simulating...")

    for timestep in range(10000000):
        #random_offset = (np.random.uniform(size=(2*envs,))-0.5)*0.08

        #if defending[1]:
        #    random_offset[1] = 0
        #else:
        #    random_offset[1] = 0.05
        with torch.no_grad():
            past_obs[:, 21] += random_offset
            past_obs[:, 25] += random_offset
            tensor_obs = TensorDict({"observation": torch.tensor(past_obs, dtype=torch.float32)}).to('cuda')

            policy_out = policy_modules[0](tensor_obs)
            actions = policy_out["action"].detach().to('cpu')
            q_value = Q_modules[0](tensor_obs)["q_value"]
            #print(q_value[1,:].detach().cpu().numpy())

            actions_np = actions.numpy()    

            def_tensor_obs = TensorDict({"observation": torch.tensor(past_obs[envs+self_play:], dtype=torch.float32)}).to('cuda')

            def_out = policy_modules[1](def_tensor_obs)
            def_actions = def_out["action"].detach().to('cpu')

            actions_np[defending & defending_mask] = def_actions[defending[envs+self_play:]]

            past_obs[:, 21] -= random_offset
            past_obs[:, 25] -= random_offset
            
        xf = np.stack([actions_np[:envs, :2], actions_np[envs:,:2]], axis=1)
        xf[:,1,:] = bounds - xf[:,1,:]

        xf[:,0,0] = np.where((xf[:,0,0] < (mallet_r + 2*puck_r + 0.01)) & (past_obs[:envs,0] < obs[:envs, 20]), mallet_r + puck_r + 0.01, xf[:,0,0])
        xf[:,0,1] = np.where((xf[:,0,1] < (mallet_r + 2*puck_r + 0.01)) & (past_obs[:envs,1] < obs[:envs, 21]), mallet_r + puck_r + 0.01, xf[:,0,1])
        xf[:,0,1] = np.where((xf[:,0,1] > (bounds[1] - mallet_r - 2*puck_r - 0.01)) & (past_obs[:envs,1] > obs[:envs, 21]), bounds[1] - mallet_r - 2*puck_r - 0.01, xf[:,0,1])

        xf[:,1,0] = np.where((xf[:,1,0] < (mallet_r + 2*puck_r + 0.01)) & (past_obs[envs:,0] < obs[envs:, 20]), mallet_r + puck_r + 0.01, xf[:,1,0])
        xf[:,1,1] = np.where((xf[:,1,1] < (mallet_r + 2*puck_r + 0.01)) & (past_obs[envs:,1] < obs[envs:, 21]), mallet_r + puck_r + 0.01, xf[:,1,1])
        xf[:,1,1] = np.where((xf[:,1,1] > (bounds[1] - mallet_r - 2*puck_r - 0.01)) & (past_obs[envs:,1] > obs[envs:, 21]), bounds[1] - mallet_r - 2*puck_r - 0.01, xf[:,1,1])

        Vo = actions_np[:,2][:, None] * Vmax * np.stack((1+actions_np[:,3], 1-actions_np[:,3]), axis=1) + np.stack((np.random.normal(0, V_std_x, (2*envs)), np.random.normal(0, V_std_y, (2*envs))), axis=-1)
        #print(actions_np[3,2])

        Vo[:,0] = np.maximum(Vo[:,0], 0.05)
        Vo[:,1] = np.maximum(Vo[:,1], 0.05)
        Vo = np.stack([Vo[:envs,:], Vo[envs:,:]], axis=1)

        rewards[:] += 0.2 * actions_np[:,4]
        no_update_mask = np.stack([actions_np[:envs,4] > np.random.random((envs,))*1.01, np.concatenate([actions_np[envs:envs+self_play,4] > np.random.random((self_play,))*1.01, actions_np[envs+self_play:,4] > np.random.random((envs-self_play,))*1.01], axis=0)], axis=1)
        action_skips += (action_skips != 0)
        action_skips += (action_skips == 0) & (actions_np[envs+self_play:,4] > 0.5)
        action_skips[action_skips == 5] = 0
        action_skips[np.logical_not(defending[envs+self_play:])] = 0

        no_update_mask[self_play:,1][action_skips != 0] = True
        xf[no_update_mask] = past_xf[no_update_mask]
        Vo[no_update_mask] = past_Vo[no_update_mask]

        past_xf = xf.copy()
        past_Vo = Vo.copy()

        sim.take_action(xf, Vo)
        sim.impulse(2/60, 0.005)

        obs[:, 24:28] = past_obs[:, 20:24]
        obs[:, 28:32] = np.concatenate([np.concatenate([xf[:,0,:], Vo[:,0,:]], axis=1), np.concatenate([bounds - xf[:,1,:], Vo[:,1,:]], axis=1)], axis=0)

        attack_mask[envs+self_play:][np.logical_not(defending[envs+self_play:])] = True

        rewards[attack_mask] += np.where((actions_np[attack_mask,0] < mallet_r+0.02) | (actions_np[attack_mask,0] > bounds[0]/2-mallet_r-0.02), -1.0, 0)
        rewards[attack_mask] += np.where((actions_np[attack_mask,1] < mallet_r+0.02) | (actions_np[attack_mask,1] > bounds[1] - mallet_r - 0.02), -1.0, 0)

        rewards[attack_mask] += np.where((actions_np[attack_mask,2] > 0.995) | (actions_np[attack_mask,2] < 0.005), -1.0, 0)
        rewards[attack_mask] += np.where((actions_np[attack_mask,3] > 0.995) | (actions_np[attack_mask,3] < -0.995), -1.0, 0)

        rewards[:] += np.where((actions_np[:,4] > 0.995) | (actions_np[:,4] < 0.005), -1.0, 0)

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
                mallet_pos, _, puck_pos, puck_vel, cL, cR = sim.step(next_img, calculate_cross_left=True)

                crossed = np.maximum(crossed, np.concatenate((cR, cL), axis=0))

                env_err = sim.check_state()
                entered_left_goal_mask, entered_right_goal_mask = sim.check_goal()
                if len(env_err) > 0:
                    print("err")
                    print(env_err)
                    for idx in env_err:
                        err_dones[idx] = True
                        if puck_pos[idx, 0] < bounds[0]/2:
                            terminal_rewards[idx] -= 100.0
                        elif idx < self_play:
                            terminal_rewards[envs+idx] -= 100.0

                terminal_rewards[:envs][entered_left_goal_mask] -= 100
                terminal_rewards[:envs][entered_right_goal_mask] += 100

                terminal_rewards[envs:envs+self_play][entered_left_goal_mask[:self_play]] += 100
                terminal_rewards[envs:][entered_right_goal_mask] -= 100

                err_dones[entered_left_goal_mask | entered_right_goal_mask] = True

                new_dones_fail = (np.logical_not(dones[envs+self_play:])) & (defending[envs+self_play:] & (puck_pos[self_play:,0] < bounds[0]/2) & (past_puck_pos[self_play:,0] > bounds[0]/2))
                new_dones_success = (np.logical_not(dones[envs+self_play:])) & (defending[envs+self_play:] & (puck_pos[self_play:,0] > bounds[0]/2) & (np.linalg.norm(puck_vel[self_play:], axis=1) < 0.5))
                dones[envs+self_play:] = dones[envs+self_play:] | new_dones_fail | new_dones_success
                terminal_rewards[envs+self_play:][new_dones_fail] += -10
                terminal_rewards[envs+self_play:][new_dones_success] += 100

                for i in range(2):
                    noise_idx = (puck_pos*100).astype(np.int16) + noise_seeds[:,i,:]
                    noise_idx[:,0] = np.clip(noise_idx[:,0], 0, 1999)
                    noise_idx[:,1] = np.clip(noise_idx[:,1], 0, 999)
                    puck_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]

                beam_interference[envs:] = np.logical_and(bounds[0]-puck_pos[:,0]+puck_r > beam_coeffs[0,0] * (bounds[0]-mallet_pos[:,1,0]) + beam_coeffs[0,1], bounds[0]-puck_pos[:,0]-puck_r < beam_coeffs[1,0] * (bounds[0]-mallet_pos[:,1,0]) + beam_coeffs[1,1])

                puck_wgn_left = np.stack((np.random.normal(0, np.where(beam_interference[:envs], puck_beam_std[0,0] * puck_pos[:,0] + puck_beam_std[0,1], puck_std[0,0] * puck_pos[:,0] + puck_std[0,1]), (envs,)), np.random.normal(0, np.where(beam_interference[:envs], puck_beam_std[1,0] * puck_pos[:,0] + puck_beam_std[1,1], puck_std[1,0] * puck_pos[:,0] + puck_std[1,1]), (envs,))),axis=1)

                puck_wgn_right = np.stack((np.random.normal(0, np.where(beam_interference[envs:], puck_beam_std[0,0] * (bounds[0]-puck_pos[:,0]) + puck_beam_std[0,1], puck_std[0,0] * (bounds[0]-puck_pos[:,0]) + puck_std[0,1]), (envs,)),np.random.normal(0, np.where(beam_interference[envs:], puck_beam_std[1,0] * (bounds[0]-puck_pos[:,0]) + puck_beam_std[1,1], puck_std[1,0] * (bounds[0]-puck_pos[:,0]) + puck_std[1,1]), (envs,))),axis=1)

                large_beam_interference[:] = np.concatenate((np.logical_and(puck_pos[:,0] > (beam_coeffs[0,0] * mallet_pos[:,0,0] + beam_coeffs[0,1]), puck_pos[:,0] < (beam_coeffs[1,0] * mallet_pos[:,0,0] + beam_coeffs[1,1])),
                    np.logical_and((bounds[0]-puck_pos[:,0]) > (beam_coeffs[0,0] * (bounds[0]-mallet_pos[:,1,0]) + beam_coeffs[0,1]), (bounds[0]-puck_pos[:,0]) < (beam_coeffs[1,0] * (bounds[0]-mallet_pos[:,1,0]) + beam_coeffs[1,1]))), axis=0)

                beam_interference[envs+self_play:] = np.logical_not(defending[envs+self_play:]) & beam_interference[envs+self_play:]
                large_beam_interference[envs+self_play:] = np.logical_not(defending[envs+self_play:]) & large_beam_interference[envs+self_play:]

                if large_beam_interference.any():
                    past_puck_data = camera_buffer.get([0])[:,:2] # player, x/y (envs,2)
                    miss_mask[:envs] = np.logical_or(np.random.random((envs)) < percent_miss[0] * puck_pos[:,0] + percent_miss[1],\
                                               np.logical_and(puck_pos[:,0] - puck_r > beam_coeffs[0,0] * mallet_pos[:,0,0] + beam_coeffs[0,1], puck_pos[:,0] + puck_r < beam_coeffs[1,0] * mallet_pos[:,0,0] + beam_coeffs[1,1]))
                    miss_mask[envs:] = np.logical_or(np.random.random((envs)) < percent_miss[0] * (bounds[0] - puck_pos[:,0]) + percent_miss[1],\
                                               np.logical_and(bounds[0] - puck_pos[:,0] - puck_r > beam_coeffs[0,0] * (bounds[0] - mallet_pos[:,1,0]) + beam_coeffs[0,1], bounds[0] - puck_pos[:,0] + puck_r < beam_coeffs[1,0] * (bounds[0] - mallet_pos[:,1,0]) + beam_coeffs[1,1]))

                    puck_pos = np.concatenate([puck_pos + puck_noise + puck_wgn_left, bounds - puck_pos - puck_noise - puck_wgn_right],axis=0)
                    puck_pos[np.logical_and(large_beam_interference, miss_mask)] = past_puck_data[np.logical_and(large_beam_interference, miss_mask)]
                else:
                    puck_pos = np.concatenate([puck_pos + puck_noise + puck_wgn_left, bounds - puck_pos - puck_noise - puck_wgn_right],axis=0)

                #mallet_noise = np.empty((envs, 4)) #mallet, op mallet
                for i in range(2,4):
                    noise_idx = (mallet_pos[:,int(i/2),:]*100).astype(np.int16) + noise_seeds[:,2+i%2,:]
                    noise_idx[:,0] = np.clip(noise_idx[:,0], 0, 1999)
                    noise_idx[:,1] = np.clip(noise_idx[:,1], 0, 999)
                    mallet_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]
                mallet_wgn_noise = np.stack((np.random.normal(0, puck_std[0,0] * mallet_pos[:,1,0] + puck_std[0,1], (envs,)), np.random.normal(0, puck_std[1,0] * mallet_pos[:,1,0] + puck_std[1,1], (envs,))), axis=-1)

                op_mallet_pos = np.concatenate([mallet_pos[:,1,:] + mallet_noise[:,2:] + mallet_wgn_noise, bounds - mallet_pos[:,0,:] - mallet_noise[:,:2]], axis=0)

                camera_buffer.put(np.concatenate([puck_pos, op_mallet_pos], axis=1))

                time_from_last_img = 0
                agent_actions.subtract(next_img)
                mallet_time.subtract(next_img)
                agent_actions.put(np.clip(np.random.normal(image_delay[0], image_delay[1]), image_delay[2], image_delay[3]))
                mallet_time.put(agent_actions.get(0) - np.clip(np.random.normal(mallet_delay[0], mallet_delay[1]), mallet_delay[2], mallet_delay[3]))

                inference_img.put(np.logical_not(inference_img.get(0)))
            elif next_mallet < next_img and next_mallet < next_action:
                mallet_pos, mallet_vel, puck_pos, puck_vel, cL, cR = sim.step(next_img, calculate_cross_left=True)

                crossed = np.maximum(crossed, np.concatenate((cR, cL), axis=0))

                env_err = sim.check_state()
                entered_left_goal_mask, entered_right_goal_mask = sim.check_goal()
                if len(env_err) > 0:
                    print("err")
                    print(env_err)
                    for idx in env_err:
                        err_dones[idx] = True
                        if puck_pos[idx, 0] < bounds[0]/2:
                            terminal_rewards[idx] -= 100.0
                        elif idx < self_play:
                            terminal_rewards[envs+idx] -= 100.0

                terminal_rewards[:envs][entered_left_goal_mask] -= 100
                terminal_rewards[:envs][entered_right_goal_mask] += 100

                terminal_rewards[envs:envs+self_play][entered_left_goal_mask[:self_play]] += 100
                terminal_rewards[envs:][entered_right_goal_mask] -= 100

                err_dones[entered_left_goal_mask | entered_right_goal_mask] = True

                new_dones_fail = (np.logical_not(dones[envs+self_play:])) & (defending[envs+self_play:] & (puck_pos[self_play:,0] < bounds[0]/2) & (past_puck_pos[self_play:,0] > bounds[0]/2))
                new_dones_success = (np.logical_not(dones[envs+self_play:])) & (defending[envs+self_play:] & (puck_pos[self_play:,0] > bounds[0]/2) & (np.linalg.norm(puck_vel[self_play:], axis=1) < 0.5))
                dones[envs+self_play:] = dones[envs+self_play:] | new_dones_fail | new_dones_success
                terminal_rewards[envs+self_play:][new_dones_fail] += -10
                terminal_rewards[envs+self_play:][new_dones_success] += 100

                mal_noise = np.random.normal(0, mallet_std, (envs,2,2))
                obs[:,20:22] = np.concatenate([mallet_pos[:,0,:] + mal_noise[:,0,:], bounds - mallet_pos[:,1,:] + mal_noise[:,1,:]], axis=0)
                vel_noise = np.random.normal(0, mallet_vel_std, size=(envs,2,2))
                obs[:,22:24] = np.concatenate([mallet_vel[:,0,:] + vel_noise[:,0,:], -mallet_vel[:,1,:] + vel_noise[:,1,:]], axis=0)
                
                time_from_last_img -= next_mallet
                agent_actions.subtract(next_mallet)
                mallet_time.subtract(next_mallet)

            elif next_action < next_img and next_action < next_mallet:
                mallet_pos, _, puck_pos, puck_vel, cL, cR = sim.step(next_img, calculate_cross_left=True)

                crossed = np.maximum(crossed, np.concatenate((cR, cL), axis=0))

                env_err = sim.check_state()
                entered_left_goal_mask, entered_right_goal_mask = sim.check_goal()
                if len(env_err) > 0:
                    print("err")
                    print(env_err)
                    for idx in env_err:
                        err_dones[idx] = True
                        if puck_pos[idx, 0] < bounds[0]/2:
                            terminal_rewards[idx] -= 100.0
                        elif idx < self_play:
                            terminal_rewards[envs+idx] -= 100.0

                terminal_rewards[:envs][entered_left_goal_mask] -= 100
                terminal_rewards[:envs][entered_right_goal_mask] += 100

                terminal_rewards[envs:envs+self_play][entered_left_goal_mask[:self_play]] += 100
                terminal_rewards[envs:][entered_right_goal_mask] -= 100

                err_dones[entered_left_goal_mask | entered_right_goal_mask] = True

                new_dones_fail = (np.logical_not(dones[envs+self_play:])) & (defending[envs+self_play:] & (puck_pos[self_play:,0] < bounds[0]/2) & (past_puck_pos[self_play:,0] > bounds[0]/2))
                new_dones_success = (np.logical_not(dones[envs+self_play:])) & (defending[envs+self_play:] & (puck_pos[self_play:,0] > bounds[0]/2) & (np.linalg.norm(puck_vel[self_play:], axis=1) < 0.5))
                dones[envs+self_play:] = dones[envs+self_play:] | new_dones_fail | new_dones_success
                terminal_rewards[envs+self_play:][new_dones_fail] += -10
                terminal_rewards[envs+self_play:][new_dones_success] += 100

                camera_obs = camera_buffer.get(indices=[img_idx, img_idx+1, img_idx+2, img_idx+5, img_idx+11])
                obs[:,:20] = camera_obs

                next_state_def[envs:] = (puck_pos[:,0] < bounds[0]/2) | (defending[envs:] & (np.linalg.norm(puck_vel, axis=1) > 0.5))
                next_state_def[:envs] = (puck_pos[:,0] > bounds[0]/2) | (defending[:envs] & (np.linalg.norm(puck_vel, axis=1) > 0.5))
                camera_obs = camera_buffer.get(indices=[img_idx+img_offset, img_idx+img_offset+1, img_idx+img_offset+2, img_idx+img_offset+5, img_idx+img_offset+11])[envs+self_play:][next_state_def[envs+self_play:]]
                obs[envs+self_play:,:20][next_state_def[envs+self_play:]] = camera_obs

                time_from_last_img -= next_action
                agent_actions.subtract(next_action)
                mallet_time.subtract(next_action)

                break

        actions_since_reset += 1
        attack_mask[envs+self_play:] = np.logical_not(defending[envs+self_play:]) & np.logical_not(next_state_def[envs+self_play:])

        rewards[:envs] += np.where((puck_pos[:,0] < bounds[0]/2) & (puck_vel[:,0] == 0) & (puck_vel[:,1] == 0), -0.8, 0)
        rewards[envs:] += np.where((puck_pos[:,0] > bounds[0]/2) & (puck_vel[:,0] == 0) & (puck_vel[:,1] == 0), -0.8, 0)

        rewards[:envs][np.logical_not(defending[:envs]) & (puck_pos[:,0] > bounds[0]/2)] += (np.linalg.norm(puck_vel,axis=1)*6 * crossed[:envs])[np.logical_not(defending[:envs]) & (puck_pos[:,0] > bounds[0]/2)]
        rewards[envs:][np.logical_not(defending[envs:]) & (puck_pos[:,0] < bounds[0]/2)] += (np.linalg.norm(puck_vel,axis=1)*6 * crossed[envs:])[np.logical_not(defending[envs:]) & (puck_pos[:,0] < bounds[0]/2)]

        rewards[:envs][(past_puck_pos[:,0] < bounds[0]/2) & (puck_pos[:,0] > bounds[0]/2)] -= 5.0
        rewards[envs:][(past_puck_pos[:,0] > bounds[0]/2) & (puck_pos[:,0] < bounds[0]/2)] -= 5.0

        rewards[:envs][(past_puck_pos[:,0] < bounds[0]/2) & (puck_pos[:,0] > bounds[0]/2) & (np.linalg.norm(puck_vel,axis=1) < 5.0)] -= 10.0
        rewards[envs:][(past_puck_pos[:,0] > bounds[0]/2) & (puck_pos[:,0] < bounds[0]/2) & (np.linalg.norm(puck_vel,axis=1) < 5.0)] -= 10.0

        rewards[:envs][(past_puck_pos[:,0] < bounds[0]/2) & (puck_pos[:,0] > bounds[0]/2) & (np.linalg.norm(puck_vel,axis=1) < 2.0)] -= 20.0
        rewards[envs:][(past_puck_pos[:,0] > bounds[0]/2) & (puck_pos[:,0] < bounds[0]/2) & (np.linalg.norm(puck_vel,axis=1) < 2.0)] -= 20.0
        #rewards[:envs][defending[:envs] & (past_puck_pos[:envs,0] < (bounds[0]/2+2*puck_r)) & (puck_pos[:envs,0] > (bounds[0]/2)+2*puck_r)] -= 10.0
        rewards[:envs+self_play][defending[:envs+self_play] & np.logical_not(next_state_def[:envs+self_play])] += 5.0

        obs[:,38] = next_state_def[:]
        dones[:envs] = err_dones #| (defending[envs:] & np.logical_not(next_state_def[envs:]))
        dones[envs:] = dones[envs:] | err_dones
        #dones[envs:envs+self_play] = dones[envs:envs+self_play] | (defending[:self_play] & np.logical_not(next_state_def[:self_play]))

        rewards[dones] += terminal_rewards[dones]
        rewards /= 25
        terminal_rewards[dones] = 0

        #print(np.linalg.norm(puck_vel[1]))

        #if not defending[envs+self_play] and not next_state_def[envs+self_play]:
        #print("----")
        #print(past_obs[0])
        #print(actions_np[0])
        #if abs(rewards[1]) > 0.01:
        #print(defending[1])
        """
        if (crossed[1] != 0) or (crossed[envs+1] != 0):
        
        #
            print("---")
            print(crossed[1])
            
            #print(defending[1])
            print(rewards[1])
            print(np.linalg.norm(puck_vel,axis=1)[1])
            print(crossed[envs+1])
            print(rewards[envs+1])
            print(defending[1])
            print(defending[envs+1])
            print(timestep)

        if (np.logical_not(defending[:envs]) & (puck_pos[:,0] > bounds[0]/2))[1] or (np.logical_not(defending[envs:]) & (puck_pos[:,0] < bounds[0]/2))[1]:
            print("_-_-_-")
            print(crossed[1])
            print(crossed[envs+1])
            print(timestep)
        """
        if abs(rewards[1]) > 0.5:
            print(rewards[1])
        #print(obs[0])
        #print(dones[0])

        defending[:] = next_state_def

        sim.display_state(1)
        print(np.linalg.norm(puck_vel[1,:]))

        dones[:envs] = err_dones
        dones[envs:] = err_dones

        num_resets = int(np.sum(dones[:envs]))

        if num_resets > 0:
            ab_vars = np.random.uniform(
                    low=ab_ranges[:, 0].reshape(1, 1, 6),    # Shape: (1, 1, 5)
                    high=ab_ranges[:, 1].reshape(1, 1, 6),   # Shape: (1, 1, 5)
                    size=(num_resets, 2, 6)                     # Output shape
                )

            ab_vars[:,:,3:] = ab_vars[:,:,3:] / np.abs(ab_vars[:,:,3:]) * np.minimum(np.abs(ab_vars[:,:,3:]), np.abs(ab_vars[:,:,:3])*9.5/10)

            ab_vars *= np.random.uniform(speed_var[0], speed_var[1], size=(num_resets, 2, 1))

            ab_obs = np.zeros((2*num_resets, 6))
                
            ab_obs[:num_resets, :] = (ab_vars[:,0,:] / pullyR) * ab_obs_scaling[:]
            ab_obs[num_resets:, :] = (ab_vars[:,1,:] / pullyR) * ab_obs_scaling[:]

            obs[dones] = np.concatenate([obs_init[dones], ab_obs, def_init[dones]], axis=1)

            camera_buffer.reset(np.where(dones)[0])

            sim.reset_sim(np.where(dones[:envs])[0], ab_vars)

            defending[envs:][dones[:envs]] = True
            defending[:envs][dones[:envs]] = False

        past_obs = obs.copy()
        dones[:] = False
        rewards[:] = 0
        err_dones[:] = False
        past_puck_pos[:] = puck_pos
        crossed[:] = 0
