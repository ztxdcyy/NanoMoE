import os
import torch
import torch.distributed as dist
from all2all.reference import generate_input, check_implementation
from all2all.submission import custom_kernel

args = dict(num_experts=8, experts_per_token=2, hidden_dim=16,
            max_num_tokens=4, seed=123, rank=0, world_size=1)
data = generate_input(**args)

if dist.is_available() and not dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=args["rank"], world_size=args["world_size"])

# output：custom_kernel(data)
# input：data
# check_implementation 对比
ok, msg = check_implementation(data, custom_kernel(data))
print("pass?", ok, msg)
