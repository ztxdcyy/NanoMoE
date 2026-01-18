import argparse
import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from layers.ep_moe import EPMoE, EPMoEConfig  # noqa: E402


def _sync_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _broadcast_state_dict(model: torch.nn.Module, src: int = 0):
    for t in model.state_dict().values():
        dist.broadcast(t, src=src)


def _build_model(cfg: EPMoEConfig, device: torch.device, rank: int, world_size: int, mode: str) -> EPMoE:
    gate = torch.nn.Linear(cfg.hidden_dim, cfg.num_experts, bias=False).to(device)
    experts = torch.nn.ModuleList(
        [torch.nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False).to(device) for _ in range(cfg.num_experts // world_size)]
    )
    return EPMoE(cfg, gate=gate, experts=experts, rank=rank, world_size=world_size, compute_mode=mode).to(device)


def _time_forward(model: EPMoE, x: torch.Tensor, warmup: int, iters: int) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        _sync_device()
        dist.barrier()
        if x.is_cuda:
            torch.cuda.profiler.start()
        start = time.perf_counter()
        for _ in range(iters):
            model(x)
        _sync_device()
        end = time.perf_counter()
        if x.is_cuda:
            torch.cuda.profiler.stop()
        dist.barrier()
        # 对比e2e执行时间
    return (end - start) / iters


def _bench_mode(mode: str, args, rank: int, world_size: int, device: torch.device, base_state=None):
    cfg = EPMoEConfig(
        num_experts=args.num_experts,
        experts_per_token=args.experts_per_token,
        hidden_dim=args.hidden_dim,
        max_num_tokens=args.max_num_tokens,
        compute_mode=mode,  # type: ignore[arg-type]
    )
    model = _build_model(cfg, device, rank, world_size, mode)
    if base_state is not None:
        model.load_state_dict(base_state)
    _broadcast_state_dict(model, src=0)

    x = torch.randn(args.batch_size, args.hidden_dim, device=device, dtype=torch.float32)
    elapsed = _time_forward(model, x, args.warmup, args.iters) if args.cuda_profiler else _time_forward_no_profiler(model, x, args.warmup, args.iters)

    t = torch.tensor([elapsed], device=device)
    dist.reduce(t, dst=0, op=dist.ReduceOp.MAX)
    return t.item()


def _time_forward_no_profiler(model: EPMoE, x: torch.Tensor, warmup: int, iters: int) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        _sync_device()
        dist.barrier()
        start = time.perf_counter()
        for _ in range(iters):
            model(x)
        _sync_device()
        end = time.perf_counter()
        dist.barrier()
    return (end - start) / iters


def _worker(rank: int, args):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.master_port))

    backend = args.backend
    use_cuda = backend == "nccl" and torch.cuda.is_available()
    device = torch.device(f"cuda:{rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(rank)

    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=args.world_size)

    try:
        torch.manual_seed(args.seed)
        base_cfg = EPMoEConfig(
            num_experts=args.num_experts,
            experts_per_token=args.experts_per_token,
            hidden_dim=args.hidden_dim,
            max_num_tokens=args.max_num_tokens,
            compute_mode="loop",
        )
        base_state = _build_model(base_cfg, device, rank, args.world_size, "loop").state_dict()

        results = {}
        modes = ["loop", "batched_gemm"] if args.mode == "both" else [args.mode]

        for mode in modes:
            t = _bench_mode(mode, args, rank, args.world_size, device, base_state=base_state)
            results[mode] = t

        if rank == 0:
            for mode in modes:
                print(f"[mode={mode}] max_rank_time = {results[mode]*1000:.3f} ms/iter "
                      f"(warmup {args.warmup}, iters {args.iters}, backend {backend})")
    finally:
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-card EPMoE/ATA benchmark")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="默认使用 num_experts * tokens_per_expert")
    parser.add_argument("--tokens-per-expert", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--experts-per-token", type=int, default=8)
    parser.add_argument("--max-num-tokens", type=int, default=None,
                        help="默认与 batch-size 相同")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--backend", type=str, default="nccl" if torch.cuda.is_available() else "gloo")
    parser.add_argument("--mode", choices=["loop", "batched_gemm", "both"], default="both",
                        help="选择单个模式或同时对比")
    parser.add_argument("--master-port", type=int, default=29523)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--cuda-profiler",
        action="store_true",
        help="Wrap measured iters with torch.cuda.profiler.{start,stop} for nsys capture-range=cudaProfilerApi.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size is None:
        args.batch_size = args.num_experts * args.tokens_per_expert
    if args.max_num_tokens is None:
        args.max_num_tokens = args.batch_size
    mp.spawn(_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
