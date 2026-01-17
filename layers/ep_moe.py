import argparse
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.cuda.nvtx as nvtx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from all2all.submission import PyTorchAllToAll

ComputeMode = Literal["loop", "batched_gemm"]


@dataclass
class EPMoEConfig:
    num_experts: int
    experts_per_token: int
    hidden_dim: int
    max_num_tokens: int
    in_dtype: torch.dtype = torch.float16
    out_dtype: torch.dtype = torch.float16
    compute_mode: ComputeMode = "loop"


class EPMoE(nn.Module):
    """
    Expert-Parallel MoE layer with a pluggable expert compute kernel.

    Supports:
    - Single GPU fast path that skips all-to-all.
    - Multi-GPU path that relies on the PyTorchAllToAll dispatcher.
    - Two expert compute modes: per-expert loop and batched GEMM.
    """

    def __init__(
        self,
        cfg: EPMoEConfig,
        gate: nn.Module | None = None,
        experts: Sequence[nn.Module] | None = None,
        rank: int = 0,
        world_size: int = 1,
        compute_mode: ComputeMode | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.compute_mode: ComputeMode = (
            (compute_mode or cfg.compute_mode or "loop").lower()  # type: ignore[assignment]
        )  # type: ignore[assignment]
        if self.compute_mode not in {"loop", "batched_gemm"}:
            raise ValueError("compute_mode must be 'loop' or 'batched_gemm'")

        self.gate = gate or nn.Linear(cfg.hidden_dim, cfg.num_experts, bias=False)

        local_experts = cfg.num_experts // world_size
        if experts is None:
            experts = [nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False) for _ in range(local_experts)]
        if len(experts) != local_experts:
            raise ValueError(f"expected {local_experts} local experts, got {len(experts)}")
        self.experts = nn.ModuleList(experts)

        self.ata = PyTorchAllToAll(cfg, rank, world_size)
        self._cached_weight = None

    def _prepare_single_card(self, x: torch.Tensor, indices: torch.Tensor):
        token_map = [[] for _ in range(self.ata.num_local_experts)]
        for t, expert_list in enumerate(indices.tolist()):
            for k, e in enumerate(expert_list):
                local_eid = e % self.ata.num_local_experts
                token_map[local_eid].append((t, k, e))

        expert_num = torch.tensor([len(lst) for lst in token_map], device=x.device, dtype=torch.int32)
        expert_meta = torch.zeros(
            (self.ata.num_local_experts, self.ata.max_recv, self.ata.META_DIM),
            device=x.device,
            dtype=torch.int32,
        )
        expert_inputs = []
        for local_eid, lst in enumerate(token_map):
            for pos, (t, k, e) in enumerate(lst):
                expert_meta[local_eid, pos, 0] = e
                expert_meta[local_eid, pos, 1] = self.rank
                expert_meta[local_eid, pos, 2] = t
                expert_meta[local_eid, pos, 3] = k
            idx = [t for t, _, _ in lst]
            expert_inputs.append(x[idx] if idx else None)

        return expert_num, expert_meta, expert_inputs

    def _compute_loop(
        self,
        expert_num: torch.Tensor,
        expert_inputs: Iterable[torch.Tensor | None],
        expert_x: torch.Tensor | None,
        expert_y: torch.Tensor,
    ):
        for local_eid in range(self.ata.num_local_experts):
            cnt = int(expert_num[local_eid].item())
            if cnt == 0:
                continue
            if self.world_size == 1:
                x_slice = expert_inputs[local_eid].to(torch.float32)  # type: ignore[index]
            else:
                x_slice = expert_x[local_eid, :cnt].to(torch.float32)  # type: ignore[index]
            y_slice = self.experts[local_eid](x_slice).to(self.cfg.out_dtype)       # 在loop内进行专家计算，kernel-launch开销较大
            expert_y[local_eid, :cnt] = y_slice
        return expert_y

    def _compute_batched_gemm(
        self,
        expert_num: torch.Tensor,
        expert_inputs: Iterable[torch.Tensor | None],
        expert_x: torch.Tensor | None,
        expert_y: torch.Tensor,
        weight_stack: torch.Tensor,
    ):
        max_cnt = int(expert_num.max().item()) if expert_num.numel() > 0 else 0
        if max_cnt == 0:
            return expert_y

        num_local = self.ata.num_local_experts
        x_batch = torch.zeros(
            (num_local, max_cnt, self.cfg.hidden_dim),
            device=expert_y.device,
            dtype=torch.float32,
        )
        for local_eid in range(num_local):
            cnt = int(expert_num[local_eid].item())
            if cnt == 0:
                continue
            if self.world_size == 1:
                x_slice = expert_inputs[local_eid].to(torch.float32)  # type: ignore[index]
            else:
                x_slice = expert_x[local_eid, :cnt].to(torch.float32)  # type: ignore[index]
            x_batch[local_eid, :cnt] = x_slice

        y_batch = torch.bmm(x_batch, weight_stack.transpose(1, 2))      # 调用一个bmm完成全部专家推理，在计算量更大的场景有显著收益

        y_batch = y_batch.to(self.cfg.out_dtype)
        for local_eid in range(num_local):
            cnt = int(expert_num[local_eid].item())
            if cnt:
                expert_y[local_eid, :cnt] = y_batch[local_eid, :cnt]
        return expert_y

    def _get_batched_params(self):
        # 仅在 eval 时缓存，训练或权重更新时跳过缓存。
        if self.training or self._cached_weight is None:
            weights = []
            for expert in self.experts:
                if not isinstance(expert, nn.Linear):
                    raise TypeError("batched_gemm requires nn.Linear experts")
                weights.append(expert.weight.to(torch.float32))
            weight_stack = torch.stack(weights, dim=0)
            if not self.training:
                self._cached_weight = weight_stack
            return weight_stack
        return self._cached_weight

    def _combine_single(
        self,
        out_tokens: torch.Tensor,
        weights: torch.Tensor,
        expert_meta: torch.Tensor,
        expert_y: torch.Tensor,
        expert_num: torch.Tensor,
    ) -> torch.Tensor:
        # 本地累加，不经过通信，便于单卡推理或无进程组场景。
        acc = torch.zeros_like(out_tokens, dtype=torch.float32)
        for local_eid in range(self.ata.num_local_experts):
            cnt = int(expert_num[local_eid].item())
            for j in range(cnt):
                meta = expert_meta[local_eid, j]
                src_token = int(meta[2].item())
                src_k = int(meta[3].item())
                w = weights[src_token, src_k].to(torch.float32)
                acc[src_token] += expert_y[local_eid, j].to(torch.float32) * w
        return acc.to(out_tokens.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] > self.cfg.max_num_tokens:
            raise ValueError("batch size exceeds cfg.max_num_tokens")

        cfg = self.cfg
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, cfg.experts_per_token, dim=-1)
        indices = indices.to(torch.int64)
        weights = weights.to(torch.float32)

        if self.world_size == 1:
            with nvtx.range("dispatch_single"):
                expert_num, expert_meta, expert_inputs = self._prepare_single_card(x, indices)
            expert_x = None
        else:
            with nvtx.range("dispatch_all2all"):
                expert_num, expert_x, expert_meta = self.ata.dispatch(x.to(cfg.in_dtype), indices)
            expert_inputs = None

        expert_y = torch.zeros(
            (self.ata.num_local_experts, self.ata.max_recv, cfg.hidden_dim),
            device=x.device,
            dtype=cfg.out_dtype,
        )

        if self.compute_mode == "loop":
            with nvtx.range("compute_loop"):
                expert_y = self._compute_loop(expert_num, expert_inputs, expert_x, expert_y)  # type: ignore[arg-type]
        else:
            weight_stack = self._get_batched_params()
            with nvtx.range("compute_batched_gemm"):
                expert_y = self._compute_batched_gemm(  # type: ignore[arg-type]
                    expert_num, expert_inputs, expert_x, expert_y, weight_stack
                )

        out_tokens = torch.zeros(cfg.max_num_tokens, cfg.hidden_dim, device=x.device, dtype=cfg.out_dtype)

        if self.world_size == 1:
            with nvtx.range("combine_single"):
                out_tokens = self._combine_single(out_tokens, weights, expert_meta, expert_y, expert_num)
        else:
            if not dist.is_initialized():
                raise RuntimeError("world_size>1 需要先调用 dist.init_process_group，再执行前向。")
            with nvtx.range("combine_all2all"):
                out_tokens = self.ata.combine(out_tokens, weights, expert_meta, expert_y, expert_num)

        out_tokens = out_tokens[: x.shape[0]]
        return out_tokens


def _parse_args():
    parser = argparse.ArgumentParser(description="EP MoE forward demo")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--experts-per-token", type=int, default=2)
    parser.add_argument("--max-num-tokens", type=int, default=8)
    parser.add_argument(
        "--compute-mode",
        choices=["loop", "batched_gemm"],
        default=os.environ.get("MOE_COMPUTE_MODE", "loop"),
        help="Choose MoE expert compute path",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--world-size", type=int, default=1)
    return parser.parse_args()


def _main():
    args = _parse_args()
    device = torch.device(args.device)

    cfg = EPMoEConfig(
        num_experts=args.num_experts,
        experts_per_token=args.experts_per_token,
        hidden_dim=args.hidden_dim,
        max_num_tokens=args.max_num_tokens,
        in_dtype=torch.float16,
        out_dtype=torch.float16,
        compute_mode=args.compute_mode,  # type: ignore[arg-type]
    )
    gate = nn.Linear(cfg.hidden_dim, cfg.num_experts, bias=False).to(device)
    experts = nn.ModuleList(
        [nn.Linear(cfg.hidden_dim, cfg.hidden_dim).to(device) for _ in range(cfg.num_experts // args.world_size)]
    )

    moe = EPMoE(cfg, gate=gate, experts=experts, rank=0, world_size=args.world_size, compute_mode=args.compute_mode)
    moe.to(device)
    moe.eval()

    x = torch.randn(args.batch_size, cfg.hidden_dim, device=device, dtype=torch.float32)
    with torch.no_grad():
        y = moe(x)
    print(f"compute_mode={args.compute_mode}, output shape={tuple(y.shape)}")


if __name__ == "__main__":
    _main()
