import torch
import torch.nn as nn
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, MLP, QValueActor, EGreedyWrapper, SafeProbabilisticModule
import numpy as np
import torch.nn.functional as F
from torchrl.data.tensor_specs import BoundedTensorSpec, CompositeSpec
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule
import tensordict


load_filepath = "checkpoints/model_183.pth"
save_filepath = "checkpoints/model_184.pth"

height = 1.9885
width = 0.9905
bounds = np.array([height, width])
mallet_r = 0.05082

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

obs_dim = 51
action_dim = 4
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


low = torch.tensor([mallet_r+0.01, mallet_r+0.01, 0, -1], dtype=torch.float32)
high = torch.tensor([bounds[0]/2-mallet_r-0.01, bounds[1] - mallet_r-0.01, 1, 1], dtype=torch.float32)


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
)


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
)

if load_filepath is not None:
    checkpoint = torch.load(load_filepath, map_location='cuda')
    policy_module.load_state_dict(checkpoint['policy_state_dict'])
    value_module.load_state_dict(checkpoint['value_state_dict'])


insert_indices = [39, 47, 53]  # where the flags go

policy_state_dict = policy_module.state_dict()
value_state_dict = value_module.state_dict()

weight0_policy = policy_state_dict['module.0.module.0.weight'].numpy()
weight0_value = value_state_dict['module.0.weight'].numpy()
print(weight0_value.shape)

# Create new array of zeros
new_weight0_policy = np.zeros((1024, 54), dtype=weight0_policy.dtype)
new_weight0_value = np.zeros((1024, 54), dtype=weight0_value.dtype)

# Figure out where to copy original columns
mask = np.ones((54,), dtype=bool)
mask[insert_indices] = False  # False where zeros should stay

# Fill remaining positions with the original data
new_weight0_policy[:, mask] = weight0_policy
new_weight0_value[:, mask] = weight0_value

policy_state_dict['module.0.module.0.weight'] = torch.tensor(new_weight0_policy, dtype=torch.float32)
value_state_dict['module.0.weight'] = torch.tensor(new_weight0_value, dtype=torch.float32)


torch.save({
    'policy_state_dict': policy_state_dict,
    'value_state_dict': value_state_dict,
}, save_filepath)
