import os, torch, torch.distributed as dist
from all2all.reference import generate_input, check_implementation
from all2all.submission import custom_kernel

rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world, device_id=rank)

args = dict(num_experts=8, experts_per_token=2, hidden_dim=16,
            max_num_tokens=4, seed=123, rank=rank, world_size=world)
data = generate_input(**args)
ok, msg = check_implementation(data, custom_kernel(data))
print(f"rank{rank} pass? {ok} {msg}")
dist.destroy_process_group()
