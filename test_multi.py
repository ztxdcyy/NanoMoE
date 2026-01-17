import os, torch, torch.distributed as dist
from all2all.reference import generate_input, check_implementation
from all2all.submission import custom_kernel

# torchrun sets LOCAL_RANK / RANK / WORLD_SIZE automatically.
rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
world = int(os.environ.get("WORLD_SIZE", "1"))

if dist.is_available() and not dist.is_initialized():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world)

args = dict(num_experts=8, experts_per_token=2, hidden_dim=16,
            max_num_tokens=4, seed=123, rank=rank, world_size=world)
data = generate_input(**args)
ok, msg = check_implementation(data, custom_kernel(data))
print(f"rank{rank} pass? {ok} {msg}")
dist.destroy_process_group()
