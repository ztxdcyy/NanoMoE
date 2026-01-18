import argparse
import os
import sys
import time
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from layers.ep_moe import EPMoE, EPMoEConfig


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark EPMoE loop vs batched_gemm")
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
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mode", choices=["loop", "batched_gemm", "both"], default="both")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--world-size", type=int, default=1, help="Benchmark currently assumes single rank")
    parser.add_argument(
        "--cuda-profiler",
        action="store_true",
        help="Wrap measured iters with torch.cuda.profiler.{start,stop} for nsys capture-range=cudaProfilerApi.",
    )
    return parser.parse_args()


def _maybe_cuda_profiler_start(enabled: bool, x: torch.Tensor):
    if enabled and x.is_cuda:
        torch.cuda.profiler.start()


def _maybe_cuda_profiler_stop(enabled: bool, x: torch.Tensor):
    if enabled and x.is_cuda:
        torch.cuda.profiler.stop()


def time_forward(moe: EPMoE, x: torch.Tensor, warmup: int, iters: int, cuda_profiler: bool) -> float:
    moe.eval()
    with torch.no_grad():
        for _ in range(warmup):
            moe(x)
        _sync()
        _maybe_cuda_profiler_start(cuda_profiler, x)
        start = time.perf_counter()
        for _ in range(iters):
            moe(x)
        _sync()
        end = time.perf_counter()
        _maybe_cuda_profiler_stop(cuda_profiler, x)
    return (end - start) / iters


def main():
    args = parse_args()
    if args.world_size != 1:
        raise ValueError("当前基准仅支持单卡运行")
    if args.batch_size is None:
        args.batch_size = args.num_experts * args.tokens_per_expert
    if args.max_num_tokens is None:
        args.max_num_tokens = args.batch_size

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    x = torch.randn(args.batch_size, args.hidden_dim, device=device, dtype=torch.float32)

    if args.mode in {"loop", "both"}:
        cfg_loop = EPMoEConfig(
            num_experts=args.num_experts,
            experts_per_token=args.experts_per_token,
            hidden_dim=args.hidden_dim,
            max_num_tokens=args.max_num_tokens,
            compute_mode="loop",
        )
        moe_loop = EPMoE(cfg_loop, rank=0, world_size=args.world_size, compute_mode="loop").to(device)
    else:
        moe_loop = None

    if args.mode in {"batched_gemm", "both"}:
        cfg_bmm = EPMoEConfig(
            num_experts=args.num_experts,
            experts_per_token=args.experts_per_token,
            hidden_dim=args.hidden_dim,
            max_num_tokens=args.max_num_tokens,
            compute_mode="batched_gemm",
        )
        moe_bmm = EPMoE(cfg_bmm, rank=0, world_size=args.world_size, compute_mode="batched_gemm").to(device)
    else:
        moe_bmm = None

    if args.mode == "both":
        assert moe_loop is not None and moe_bmm is not None
        moe_bmm.load_state_dict(moe_loop.state_dict())
        with torch.no_grad():
            y_loop = moe_loop(x)
            y_bmm = moe_bmm(x)
            max_err = (y_loop - y_bmm).abs().max().item()
            ok = torch.allclose(y_loop, y_bmm, rtol=1e-3, atol=1e-3)
        print(f"一致性检查: {'PASS' if ok else 'FAIL'}, max_err={max_err:.4e}")

    if moe_loop is not None:
        t_loop = time_forward(moe_loop, x, args.warmup, args.iters, args.cuda_profiler)
        print(f"loop        : {t_loop*1000:.3f} ms/iter (warmup {args.warmup}, iters {args.iters})")

    if moe_bmm is not None:
        t_bmm = time_forward(moe_bmm, x, args.warmup, args.iters, args.cuda_profiler)
        print(f"batched_gemm: {t_bmm*1000:.3f} ms/iter (warmup {args.warmup}, iters {args.iters})")


if __name__ == "__main__":
    main()
