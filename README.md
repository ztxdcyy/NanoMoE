MoE推理优化集成项目

- [ ]  nanomoe ep
    - [ ]  moe（mlp）计算新增 backend
    - [ ]  使用batchgemm < fusedmoe（triton）< deepgemm
    - [ ]  attn计算使用mla（triton）
    - [ ]  通信上使用deepep
    - [ ]  整体上使用tbo
- [ ]  eplb
- [ ]  shared-expert-overlap

```
单卡基准（CPU/GPU均可）：

python bench_ep_moe.py --batch-size 8 --hidden-dim 256 --warmup 5 --iters 20

多卡基准（需多 GPU 或 gloo 多进程）：

python bench_ep_moe_multi.py --world-size 2 --backend nccl --mode both --batch-size 8 --hidden-dim 256 --warmup 5 --iters 20（如端口冲突可加 --master-port）。
```

# 记录

## bmm

在简单的将循环替换成bmm之后，并没有获得预期的收益：
```
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
```


```
(base) root@autodl-container-f99d409ef7-af0a6bc9:~/workspace/NanoMoE# python bench/bench_ep_moe_multi.py --world-size 2 --backend nccl --batch-size 128
[mode=loop] max_rank_time = 134.971 ms/iter (warmup 5, iters 20, backend nccl)
[mode=batched_gemm] max_rank_time = 132.881 ms/iter (warmup 5, iters 20, backend nccl)
```

跑了nsys之后，可以直观的看到，通信开销远大于计算开销：
* Top 两个 kernel 都是 NCCL：ncclDevKernel_Broadcast_RING_LL（61%）和 ncclDevKernel_SendRecv（12.5%）。说明多数时间花在通信上，而且使用的是 Ring/LL 协议（LL 典型用于小包，延迟为主）。
* 计算相关的只占少数：若干 add/copy（各 ~5%）、小矩阵 GEMM（ampere_sgemm_64x32... 等合计不到 10%）。整体看计算占比很低，通信成为瓶颈。

```
 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     61.4        678424202        516  1314775.6  1287769.0    168706  19288827     797388.1  ncclDevKernel_Broadcast_RING_LL(ncclDevComm *, unsigned long, ncclWork *)                           
     12.5        138628456        168   825169.4    69920.0      4832  10338753    1959357.5  ncclDevKernel_SendRecv(ncclDevComm *, unsigned long, ncclWork *)                                    
      5.2         57509588      28672     2005.8     2016.0      1952      2528         31.2  void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<float>, at::detail::Array<…
      5.2         57045483      32179     1772.8     1696.0      1632      3136        200.7  void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIterator…
      3.5         38307042       1715    22336.5    21472.0     20512     35584       2136.8  ampere_sgemm_64x32_sliced1x4_tn                                                                     
      3.3         36868709      28672     1285.9     1280.0      1248      1536         25.1  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::n…
      2.3         24922783          7  3560397.6  3556539.0   3552508   3583932      10781.2  ampere_sgemm_128x128_tn                                                                             
      1.5         17099855          7  2442836.4  2443279.0   2440208   2443984       1369.7  void gemmSN_TN_kernel<float, (int)128, (int)16, (int)2, (int)4, (int)8, (int)9, (bool)0, cublasGemv…
```