import torch
from all2all.reference import generate_input, check_implementation
from all2all.submission import custom_kernel

args = dict(num_experts=8, experts_per_token=2, hidden_dim=16,
            max_num_tokens=4, seed=123, rank=0, world_size=1)
data = generate_input(**args)
ok, msg = check_implementation(data, custom_kernel(data))
print("pass?", ok, msg)