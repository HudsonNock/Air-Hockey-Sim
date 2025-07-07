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
from perlin_numpy import generate_perlin_noise_2d


# Simulation parameters
obs_dim = 51 #[puck_pos, opp_mallet_pos] #guess mean low std
            #[mallet_pos, mallet_vel, past action, past mallet_pos]
            #[e_n0, e_nr, e_nf, e_t0, e_tr, e_tf, var] x2 mallet and wall,
            #[a1, a2, a3, b1, b2]

mallet_r = 0.05082
puck_r = 0.5 * 0.0618
pullyR = 0.035306

#col: n_f + (1-n_f/n_0) * 2/(1+e^(n_r x^2)) n_0

action_dim = 4
Vmax = 24

sensor_error = 0.0027 #mean of 1mm between [0-2.5 mm]

#mean, var, min, max
camera_to_laptop_delay = [0.03, 0.0003, 0.03-0.0004, 0.03+0.0004]
image_and_inference_delay = [0.006, 0.0014, 0.0045, 0.009]
serial_delay = [0.001, 0.0002, 0.0002, 0.0015]

frame_interval = [0.0085, 0.0001, 0.0084, 0.0086]
frames = [0, 1, 2, 4, 9]

height = 1.9885
width = 0.9905
bounds = np.array([height, width])
goal_width = 0.254

class ScaledNormalParamExtractor(NormalParamExtractor):
    def __init__(self, scale_factor=1.0):
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
    ScaledNormalParamExtractor(),
).to('cuda')

policy_module = TensorDictModule(
    policy_net, in_keys=["observation"], out_keys=["loc", "scale"]
).to('cuda')

policy_module = ProbabilisticActor(
    module=policy_module,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": torch.tensor([mallet_r, mallet_r, 0.25, 0.25]),
        "high": torch.tensor([bounds[0]/2-mallet_r, bounds[1] - mallet_r, 24, 24]),  #TODO Change this since Vx + Vy < 2*Vmax * 0.8
    },
    default_interaction_type=tensordict.nn.InteractionType.RANDOM,
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
).to('cuda')

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
    nn.Linear(128, action_dim * 2),
    ScaledNormalParamExtractor(),
).to('cuda')

value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
    out_keys=["state_value"]
).to('cuda')

#policy_module.load_state_dict(torch.load("policy_weights8.pth")) #8
#value_module.load_state_dict(torch.load("value_weights8.pth"))

"""
advantage_module = GAE(
    gamma=0.99, lmbda=0.5, value_network=value_module
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=0.03, #0.01
    entropy_bonus=True, 
    entropy_coef=0.00024, #0.00024,
    )

optim = torch.optim.Adam(loss_module.parameters(), lr=4e-4) #3e-4
"""

envs = 8 #2048


# Generate 2D low-frequency Perlin noise
mallet_init = np.array([[0.25, 0.5], [0,0]])
mallet_init[1] = bounds - mallet_init[0]

puck_init = np.array([[0.5, 0.5], [0,0]])
puck_init[1] = bounds - puck_init[0]

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
    noise_idx = (puck_init[0]*100).astype(np.int16) + noise_seeds[:,i,:]
    puck_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]

mallet_noise = np.empty((envs, 4)) #mallet, op mallet
for i in range(4):
    noise_idx = (mallet_init[int(i/2)]*100).astype(np.int16) + noise_seeds[:,2+i%2,:]
    mallet_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]

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
coll_vars[:, [5, 12]] = np.random.uniform(e_nf, coll_vars[:, [3, 10]]) # e_tf

#require C6^2 > 4 * C5 * C7

#[C5x, C6x, C7x, C5y, C6y, C7y]
#[5.2955 * 10^-6, 8.449*10^-3, 
#[1.8625 * 10^-6, 2.971*10^-3, ]

#a1 = 3.579*10**(-6)
ab_ranges = np.array([
        [2.86e-6, 5.35e-6],  # a1
        [4.57e-3, 8.55e-3],  # a2
        [4.25e-2, 7.15e-2],  # a3
        [-2.19e-6, -1.38e-6], # b1 (note: corrected order for uniform sampling)
        [-3.5e-3, -2.19e-3]   # b2 (note: corrected order for uniform sampling)
    ])
#b3 is 0

ab_vars = np.random.uniform(
        low=ab_ranges[:, 0].reshape(1, 1, 5),    # Shape: (1, 1, 5)
        high=ab_ranges[:, 1].reshape(1, 1, 5),   # Shape: (1, 1, 5)
        size=(envs, 2, 5)                     # Output shape
    )

ab_vars *= np.random.uniform(0.8, 1.2, size=(envs, 2, 1))

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

sim.initalize(envs=envs, mallet_r=mallet_r, puck_r=puck_r, goal_w=goal_width, V_max=Vmax, pully_radius=pullyR, coll_vars=coll_vars, ab_vars=ab_vars)

obs[:envs, :] = np.concatenate([np.tile(np.concatenate([np.full((envs,2), puck_init[0])+puck_noise,
                                    np.full((envs,2), mallet_init[1])+mallet_noise[:,2:]], axis=1), (5,)),
                                np.full((envs,2), mallet_init[0])+mallet_noise[:,2:],
                                np.zeros((envs,2)),
                                np.full((envs,2), mallet_init[0])+mallet_noise[:,2:],
                                np.zeros((envs,2)),
                                np.full((envs,2), mallet_init[0]) + mallet_noise[:,2:],
                                np.random.rand(envs, 2) * 24,
                                coll_vars,
                                ab_obs[:envs, :]], axis=1)

obs[envs:, :] = np.concatenate([np.tile(np.concatenate([np.full((envs,2), puck_init[1])+puck_noise,
                                    np.full((envs,2), mallet_init[1])+mallet_noise[:,:2]], axis=1), (5,)),
                                np.full((envs,2), mallet_init[0])+mallet_noise[:,2:],
                                np.zeros((envs,2)),
                                np.full((envs,2), mallet_init[0])+mallet_noise[:,2:],
                                np.zeros((envs,2)),
                                np.full((envs,2), mallet_init[0]) + mallet_noise[:,2:],
                                np.random.rand(envs, 2) * 24,
                                coll_vars,
                                ab_obs[envs:, :]], axis=1)

past_obs = obs.copy()
obs_init = obs[:,:len(obs[0])-19].copy()

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
            new_indicies.append((self.head + idx-1) % self.buffer_size)
        return self.array[np.array(new_indicies)].transpose(1, 0, 2).reshape(2*envs, -1)
    
camera_buffer = np.empty((camera_buffer_size, 2*envs, 4))
camera_buffer[:, :envs, :2] = np.full((envs,2), puck_init[0])+puck_noise
camera_buffer[:, envs:, :2] = np.full((envs,2), puck_init[1])+puck_noise
camera_buffer[:, :envs, 2:] = np.full((envs,2), mallet_init[1])+mallet_noise[:,:2]
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

camera_time = np.empty((camera_buffer_size,))
camera_time[0] = np.clip(np.random.normal(frame_interval[0], frame_interval[1]), frame_interval[2], frame_interval[3])
camera_time[1] = 0

for i in range(2, len(camera_time)):
    camera_time[i] = camera_time[i-1] - np.clip(np.random.normal(frame_interval[0], frame_interval[1]), frame_interval[2], frame_interval[3])
camera_time = CircularBuffer(camera_time)

inference_img = np.zeros((camera_buffer_size,), dtype=np.bool_)
for i in range(len(inference_img)):
    if i % 2 == 1:
        inference_img[i] = True
inference_img = CircularBuffer(inference_img)

camera_send_delay = np.empty((camera_buffer_size,))
for i in range(len(camera_send_delay)):
    camera_send_delay[i] = np.clip(np.random.normal(camera_to_laptop_delay[0], camera_to_laptop_delay[1]), camera_to_laptop_delay[2], camera_to_laptop_delay[3])
camera_send_delay = CircularBuffer(camera_send_delay)

camera_arrivals = np.empty((camera_buffer_size,))
for i in range(len(camera_arrivals)):
    camera_arrivals[i] = camera_time.get(i) + camera_send_delay.get(i)
camera_arrivals = CircularBuffer(camera_arrivals)

inference_serial_delay = np.empty((camera_buffer_size))
for i in range(len(inference_serial_delay)):
    inference_serial_delay[i] = np.clip(np.random.normal(image_and_inference_delay[0], image_and_inference_delay[1]),image_and_inference_delay[2], image_and_inference_delay[3]) +\
                                    np.clip(np.random.normal(serial_delay[0], serial_delay[1]),serial_delay[2], serial_delay[3])
inference_serial_delay = CircularBuffer(inference_serial_delay)

batch_size = 128
save_num = 0

#move timeline to first action time
while True:
    next_NN_inference = np.inf
    img_idx = None
    for i in range(camera_buffer_size):
        if not inference_img.get(i):
            continue
        action_time = camera_arrivals.get(i) + inference_serial_delay.get(i)
        if action_time > 1e-8 and action_time < next_NN_inference:
            next_NN_inference = action_time
            img_idx = i

    if camera_time.get(0) < next_NN_inference:
        #mallet_pos (mallet, op_mallet)
        mallet_pos, puck_pos, cross_left, cross_right = sim.step(camera_time.get(0))
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

        camera_arrivals.subtract(camera_time.get(0))
        camera_time.subtract(camera_time.get(0))
        camera_time.put(np.clip(np.random.normal(frame_interval[0], frame_interval[1]), frame_interval[2], frame_interval[3]))

        inference_img.put(np.logical_not(inference_img.get(0)))
        camera_send_delay.put(np.clip(np.random.normal(camera_to_laptop_delay[0], camera_to_laptop_delay[1]), camera_to_laptop_delay[2], camera_to_laptop_delay[3]))
        camera_arrivals.put(camera_time.get(0) + camera_send_delay.get(0))
        inference_serial_delay.put(np.clip(np.random.normal(image_and_inference_delay[0], image_and_inference_delay[1]), image_and_inference_delay[2], image_and_inference_delay[3]) +\
                                    np.clip(np.random.normal(serial_delay[0], serial_delay[1]),serial_delay[2],serial_delay[3]))
    else:
        _, _, cross_left, cross_right = sim.step(next_NN_inference)

        #[puck_pos, opp_mallet_pos] #guess mean low std
        #[mallet_pos, mallet_vel, past mallet_pos, past_mallet_vel, past action (x0, V)]
        #[e_n0, e_nr, e_nf, e_t0, e_tr, e_tf, var] x2 mallet and wall,

        # (2*envs, 20)
        camera_obs = camera_buffer.get(indices=[img_idx, img_idx+1, img_idx+2, img_idx+5, img_idx+11])
        past_obs[:,:20] = camera_obs

        camera_time.subtract(next_NN_inference)
        camera_arrivals.subtract(next_NN_inference)

        break

dones = np.zeros((2*envs,), dtype=np.bool_)
rewards = np.zeros((2*envs,))

xf_mallet_noise = np.empty((envs, 4))

if True:

    buffer_size = 300_000
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(buffer_size)
    )

    #Simulation loop
    for update in range(500000):  # Training loop
        print("simulating...")
        for timestep in range(50):

            tensor_obs = TensorDict({"observation": torch.tensor(past_obs, dtype=torch.float32)}).to('cuda')

            policy_out = policy_module(tensor_obs)
            actions = policy_out["action"].detach().to('cpu')
            policy_out["sample_log_prob"] = torch.maximum(policy_out["sample_log_prob"], torch.tensor(-8, dtype=torch.float32))
            log_prob = policy_out["sample_log_prob"].detach().to('cpu')

            if np.isnan(log_prob.numpy()).any():
                print("NAN")
                break

            actions_np = actions.numpy()

            xf = np.stack([actions_np[:envs, :2], actions_np[envs:,:2]], axis=1)
            #xf = np.full((envs,2,2), 0.3)
            
            #xf = np.random.rand(envs, 2, 2) * (0.8 - 0.2) + 0.2
            obs[:envs,28:30] = xf[:,0,:]
            obs[envs:,28:30] = xf[:,1,:]

            xf[:,1,:] = bounds - xf[:,1,:]

             #mallet, op mallet
            for i in range(4):
                noise_idx = (xf[:,int(i/2),:]*100).astype(np.int16) + noise_seeds[:,2+i%2,:]
                xf_mallet_noise[:,i] = noise_map[noise_idx[:,0], noise_idx[:,1]]
            xf[:,0,:] -= xf_mallet_noise[:,:2]
            xf[:,1,:] += xf_mallet_noise[:,2:]

            xf[:,:,0] = np.clip(xf[:,:,0], mallet_r+1e-4, bounds[0]-mallet_r-1e-4)
            xf[:,:,1] = np.clip(xf[:,:,1], mallet_r+1e-4, bounds[1]-mallet_r-1e-4)

            Vo = np.stack([actions_np[:envs, 2:], actions_np[envs:,2:]], axis=1)
            #Vo = np.full((envs, 2, 2), np.array([23, 23]))
            obs[:envs, 30:32] = Vo[:,0,:]
            obs[envs:, 30:32] = Vo[:,1,:]

            sim.take_action(xf, Vo)

            while True:
                next_NN_inference = np.inf
                img_idx = None
                for i in range(camera_buffer_size):
                    if not inference_img.get(i):
                        continue
                    action_time = camera_arrivals.get(i) + inference_serial_delay.get(i)
                    if action_time > 1e-8 and action_time < next_NN_inference:
                        next_NN_inference = action_time
                        img_idx = i

                if camera_time.get(0) < next_NN_inference:
                    #mallet_pos (mallet, op_mallet)
                    mallet_pos, puck_pos, cross_left, cross_right = sim.step(camera_time.get(0))
                    env_err = sim.check_state()
                    entered_left_goal_mask, entered_right_goal_mask = sim.check_goal()
                    if len(env_err) > 0:
                        print("err")
                        print(env_err)
                        for idx in env_err:
                            dones[idx] = True
                            dones[envs+idx] = True
                    dones[:envs][entered_left_goal_mask | entered_right_goal_mask] = True
                    dones[envs:][entered_left_goal_mask | entered_right_goal_mask] = True

                    rewards[:envs][entered_left_goal_mask] -= 10
                    rewards[:envs][entered_right_goal_mask] += 10
                    rewards[envs:][entered_left_goal_mask] += 10
                    rewards[envs:][entered_right_goal_mask] -= 10

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

                    obs_mallet_pos = mallet_pos[:,0,:] + mallet_noise[:,:2]
                    op_obs_mallet_pos = bounds - mallet_pos[:,1,:] + mallet_noise[:,2:]

                    camera_arrivals.subtract(camera_time.get(0))
                    camera_time.subtract(camera_time.get(0))
                    camera_time.put(np.clip(np.random.normal(frame_interval[0], frame_interval[1]), frame_interval[2], frame_interval[3]))

                    inference_img.put(np.logical_not(inference_img.get(0)))
                    camera_send_delay.put(np.clip(np.random.normal(camera_to_laptop_delay[0], camera_to_laptop_delay[1]), camera_to_laptop_delay[2], camera_to_laptop_delay[3]))
                    camera_arrivals.put(camera_time.get(0) + camera_send_delay.get(0))
                    inference_serial_delay.put(np.clip(np.random.normal(image_and_inference_delay[0], image_and_inference_delay[1]), image_and_inference_delay[2], image_and_inference_delay[3]) +\
                                                np.clip(np.random.normal(serial_delay[0], serial_delay[1]),serial_delay[2],serial_delay[3]))
                else:
                    _, _, cross_left, cross_right = sim.step(next_NN_inference)
                    env_err = sim.check_state()
                    entered_left_goal_mask, entered_right_goal_mask = sim.check_goal()
                    if len(env_err) > 0:
                        print("err")
                        print(env_err)
                        for idx in env_err:
                            dones[idx] = True
                            dones[envs+idx] = True
                    dones[:envs][entered_left_goal_mask | entered_right_goal_mask] = True
                    dones[envs:][entered_left_goal_mask | entered_right_goal_mask] = True

                    rewards[:envs][entered_left_goal_mask] -= 10
                    rewards[:envs][entered_right_goal_mask] += 10
                    rewards[envs:][entered_left_goal_mask] += 10
                    rewards[envs:][entered_right_goal_mask] -= 10
                    #take action
                    #[puck_pos, opp_mallet_pos] #guess mean low std
                    #[mallet_pos, mallet_vel, past mallet_pos, past_mallet_vel, past action (x0, V)]
                    #[e_n0, e_nr, e_nf, e_t0, e_tr, e_tf, var] x2 mallet and wall,

                    # (2*envs, 20)
                    camera_obs = camera_buffer.get(indices=[img_idx, img_idx+1, img_idx+2, img_idx+5, img_idx+11])

                    obs[:,:20] = camera_obs

                    camera_time.subtract(next_NN_inference)
                    camera_arrivals.subtract(next_NN_inference)

                    break
            


            replay_buffer.extend(TensorDict({
                    "observation": tensor_obs["observation"],
                    "action": torch.tensor(obs[:,28:32]),
                    "reward": torch.zeros((2*envs, 1)),
                    "done": torch.tensor(dones),
                    "next_observation": torch.tensor(obs),
                    "sample_log_prob": log_prob
                }, batch_size=[2*envs]))
            
            past_obs = obs.copy()

            sim.display_state(0)

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
                coll_vars[:, [5, 12]] = np.random.uniform(e_nf, coll_vars[:, [3, 10]]) # e_tf

                ab_vars = np.random.uniform(
                        low=ab_ranges[:, 0].reshape(1, 1, 5),    # Shape: (1, 1, 5)
                        high=ab_ranges[:, 1].reshape(1, 1, 5),   # Shape: (1, 1, 5)
                        size=(num_resets, 2, 5)                     # Output shape
                    )

                ab_vars *= np.random.uniform(0.8, 1.2, size=(num_resets, 2, 1))

                ab_obs = np.zeros((2*num_resets, 5))
                    
                ab_obs[:num_resets, :] = (ab_vars[:,0,:] / pullyR) * ab_obs_scaling[:]
                ab_obs[num_resets:, :] = (ab_vars[:,1,:] / pullyR) * ab_obs_scaling[:]

                obs[dones] = np.concatenate([obs_init[dones], np.tile(coll_vars, (2,1)), ab_obs], axis=1)

                camera_buffer.reset(np.where(dones)[0])

                sim.reset_sim(np.where(dones[:envs])[0], coll_vars, ab_vars)
                
                
            """
            cross_left = np.maximum(cross_left, cross_left2)
            cross_right = np.maximum(cross_right, cross_right2)
            puck_pos_noM, puck_vel_noM = sim.step_noM(Ts*2)

            attack_copy = np.copy(attack)

            dones = np.full((2*envs), False)
            rewards = np.zeros((2*envs))

            #Shooting reward
            dones[:envs] = np.logical_or(cross_right > 0, cross_right == -0.5)
            dones[envs:] = np.logical_or(cross_left > 0, cross_left == -0.5)

            attack_copy[dones] = False

            rewards[:envs] += np.where(cross_right > 0,\
                                 np.where(attack[:envs], np.maximum(cross_right,0)/50, np.sqrt(np.maximum(cross_right,0))/120),\
                                 np.where(np.logical_and(cross_right == -0.5, np.logical_not(attack[:envs])), -0.05, 0))
            rewards[envs:] += np.where(cross_left > 0,\
                                 np.where(attack[envs:], np.maximum(cross_left, 0)/50, np.sqrt(np.maximum(cross_left, 0))/120),\
                                 np.where(np.logical_and(cross_left == -0.5, np.logical_not(attack[envs:])), -0.05, 0))

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
            rewards -= 0.003

            vel_norm = np.linalg.norm(puck_vel, axis=1)

            stabalized_l = ~attack_copy[:envs] & left_puck & (vel_norm > 0.3) & (np.abs(puck_vel[:,1])/np.maximum(np.abs(puck_vel[:,0]), 0.001) > 2) & (np.abs(puck_pos[:,0] - 0.65) < 0.3)
            stabalized_l = stabalized_l & np.random.choice([False, True], size=envs, p=[0.3, 0.7])
            stabalized_r = ~attack_copy[envs:] & ~left_puck & (vel_norm > 0.3) & (np.abs(puck_vel[:,1])/np.maximum(np.abs(puck_vel[:,0]), 0.001) > 2) & (np.abs(puck_pos[:,0] - 1.35) < 0.3)
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

            pos_range = (-0.02, 0.02)  # Range for position (x and y)
            vel_range = (-0.06, 0.06)  # Range for velocity

            # Create random perturbations for position and velocity
            pos_perturb = np.random.uniform(pos_range[0], pos_range[1], next_obs_np[:, :2].shape)
            vel_perturb = np.random.uniform(vel_range[0], vel_range[1], next_obs_np[:, 2:4].shape)

            # Add the random perturbations to positions and velocities
            next_obs_np[:envs, :2] += pos_perturb[:envs]
            next_obs_np[:envs, 2:4] += vel_perturb[:envs]
            next_obs_np[envs:, :2] += pos_perturb[envs:]
            next_obs_np[envs:, 2:4] += vel_perturb[envs:]

            rewards = torch.tensor(rewards, dtype=torch.float32)  # Rewards
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32)  # Next observations
            terminated = torch.tensor(dones)
            dones = torch.tensor(dones) # Done flags (0 or 1)

            replay_buffer.extend(TensorDict({
                    "observation": past_obs["observation"],
                    "action": actions,
                    "reward": rewards,
                    "done": dones,
                    "next_observation": obs["observation"],
                    "terminated": terminated,
                    "sample_log_prob": log_prob
                }, batch_size=[actions.shape[0]]))

            past_obs = obs
            obs = TensorDict({"observation": next_obs})

        print("training...")
        for _ in range(250):
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
            

        torch.save(policy_module.state_dict(), "C:\\Users\\hudso\\Downloads\\policy_weights26.pth")
        torch.save(value_module.state_dict(), "C:\\Users\\hudso\\Downloads\\value_weights26.pth")
        print((update+1) / 100)
        """

    print("Done Training")

sim.initalize(envs=1, mallet_r=mallet_r, puck_r=puck_r, goal_w=goal_width, V_max=Vmax)
Ts = 0.1
envs = 1
N = 35
step_size = Ts/N
obs = np.empty((2, obs_dim))
past_obs = np.empty((2,obs_dim))
attack = np.array([True, False])
obs[:1, :] = np.concatenate([mallet_init[0], mallet_init[1], puck_init[0], np.zeros((6,)), puck_init[0], np.zeros((2,)), np.array([1.0])])
obs[1:, :] = np.concatenate([mallet_init[0], mallet_init[1], puck_init[1], np.zeros((6,)), puck_init[1], np.zeros((3,))])
past_obs[:1, :] = np.concatenate([mallet_init[0], mallet_init[1], puck_init[0], np.zeros((6,)), puck_init[0], np.zeros((2,)), np.array([1.0])])
past_obs[1:, :] = np.concatenate([mallet_init[0], mallet_init[1], puck_init[1], np.zeros((6,)), puck_init[1], np.zeros((3,))])
obs = TensorDict({"observation": torch.tensor(obs, dtype=torch.float32)}, batch_size = 2)
past_obs = TensorDict({"observation": torch.tensor(past_obs, dtype=torch.float32)}, batch_size = 2)
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

    #values = value_module(obs)["state_value"].detach().numpy()
    #print("----------")
    #print(values)
    #print(attack)
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
    #if start_cnt % 30 == 0:
    #    theta = np.random.uniform(0, 2*3.14)
    #start_cnt += 1
    theta = theta + np.random.uniform(-0.1,0.3)

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

    stabalized_l = ~attack_copy[:envs] & left_puck & (vel_norm > 0.3) & (np.abs(puck_vel[:,1])/np.maximum(np.abs(puck_vel[:,0]), 0.001) > 2) & (np.abs(puck_pos[:,0] - 0.65) < 0.3)
    #stabalized_l = stabalized_l & np.random.choice([False, True], size=envs, p=[0.3, 0.7])
    stabalized_r = ~attack_copy[envs:] & ~left_puck & (vel_norm > 0.3) & (np.abs(puck_vel[:,1])/np.maximum(np.abs(puck_vel[:,0]), 0.001) > 2) & (np.abs(puck_pos[:,0] - 1.35) < 0.3)
    #stabalized_r = stabalized_r & np.random.choice([False, True], size=envs, p=[0.3, 0.7])
    stabalized = np.concatenate([stabalized_l, stabalized_r])

    dones[stabalized] = True
    attack_copy[stabalized] = True

    rewards[:envs][stabalized_l] += 2.0
    rewards[envs:][stabalized_r] += 2.0

    #print(rewards)

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
    next_obs[envs:,:] = np.concatenate([bounds - mallet_pos[:,1,:], bounds - mallet_pos[:,0,:], bounds - puck_pos, -mallet_vel[:,1,:], -mallet_vel[:,0,:], -puck_vel, bounds - puck_pos_noM, -puck_vel_noM, attack[envs:].reshape(-1,1)], axis=1)

    pos_range = (-0.02, 0.02)  # Range for position (x and y)
    vel_range = (-0.05, 0.05)  # Range for velocity

    # Create random perturbations for position and velocity
    pos_perturb = np.random.uniform(pos_range[0], pos_range[1], next_obs[:, :2].shape)
    vel_perturb = np.random.uniform(vel_range[0], vel_range[1], next_obs[:, 2:4].shape)

    # Add the random perturbations to positions and velocities
    next_obs[:envs, :2] += pos_perturb[:envs]
    next_obs[:envs, 2:4] += vel_perturb[:envs]
    next_obs[envs:, :2] += pos_perturb[envs:]
    next_obs[envs:, 2:4] += vel_perturb[envs:]

    past_obs = obs
    obs = TensorDict({"observation": torch.tensor(next_obs, dtype=torch.float32)}, batch_size=2)

    #TODO
    """
    Remove velocity calculations, give model the puck position directly
    - Generate perlin nosie for random offset map for each env
    - Decide time interval between samples we want to give the model
    - Use same perlin noise to offset opponent mallet as well

    Have each env include randomly generated mallet movement parameters
    - Each parameter modeled with a polynomial characterising how it changes with diff voltage levels
    - Give these parameters into the model as well

    Alter how the puck moves
    - Randomize the puck motion parameters across each env (don't give parameters to the model)
    - Randomize tangential and normal restitution coefficients and add variance whenever appling them, give to model

    Adding delay
    - randomize mean and variance of the delay for each env, give the model the mean and variance
    
    """
