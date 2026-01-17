import time
import torch

def main():
    torch.manual_seed(0)

    # Configuration parameters, mapped to MoE concepts
    num_experts = 1024   # B: Number of experts (or expert computations) to process in batch
    tokens_per_expert = 64 # M: "Local batch size" or number of tokens an individual expert handles
    hidden_dim_in = 64   # K: Input hidden dimension for experts
    hidden_dim_out = 64  # N: Output hidden dimension for experts

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Randomly generate batched inputs and weights for experts
    # expert_inputs_batch[i] is (tokens_per_expert, hidden_dim_in)
    expert_inputs_batch = torch.randn(num_experts, tokens_per_expert, hidden_dim_in, device=device)
    # expert_weights_batch[i] is (hidden_dim_in, hidden_dim_out)
    expert_weights_batch = torch.randn(num_experts, hidden_dim_in, hidden_dim_out, device=device)

    # Warm-up to load CUDA kernels
    for _ in range(5):
        _ = torch.bmm(expert_inputs_batch, expert_weights_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # --------------------------------------------------------------------------
    # 1. Loop through each expert and perform individual matrix multiplication
    #    This simulates processing each expert's input with its weights one by one.
    # --------------------------------------------------------------------------
    t0 = time.perf_counter()
    expert_outputs_loop = []
    for i in range(num_experts):
        # A[i].mm(Bm[i]) -> (tokens_per_expert, hidden_dim_in) @ (hidden_dim_in, hidden_dim_out)
        expert_outputs_loop.append(expert_inputs_batch[i].mm(expert_weights_batch[i]))
    expert_outputs_loop = torch.stack(expert_outputs_loop, dim=0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    # --------------------------------------------------------------------------
    # 2. Perform batched matrix multiplication (bmm) for all experts at once
    #    This represents an optimized way to process all expert computations.
    # --------------------------------------------------------------------------
    t2 = time.perf_counter()
    expert_outputs_bmm = torch.bmm(expert_inputs_batch, expert_weights_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t3 = time.perf_counter()

    # Verify numerical equivalence (allowing for floating point differences)
    max_diff = (expert_outputs_loop - expert_outputs_bmm).abs().max().item()
    print(f"\nMax difference between looping through experts and batched MoE computation: {max_diff:.3e}")

    print(f"Individual expert computations (loop mm) time: {(t1 - t0)*1000:.2f} ms")
    print(f"Batched MoE computation (bmm) time          : {(t3 - t2)*1000:.2f} ms")
    print(f"Speedup (loop mm / bmm)                      : {(t1 - t0)/(t3 - t2):.2f}x")

if __name__ == "__main__":
    main()