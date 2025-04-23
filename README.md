# flashattention

Implementations of [Flash Attention 2](https://arxiv.org/abs/2307.08691) in CUDA. Tested on a Nvidia A10G GPU on an Amazon EC2 g5.xlarge instance.


### Worklog (optimizing with Nsight Compute)

Single head attention (using M = 10000, N = 9000, d = 32)
| Version | Optimization | Code | Duration | Compute Throughput % | Memory Throughput % | Notes |
| - | - | - | - | - | - | - |
| V1 | Baseline | [Link](./fa2_single_head_v1.cu) | 2.78s | 0.27% | 1.19% | Estimated speedup 98.75% since only 1 of 80 SMs being used
| V2 | Parallelized work over multiple thread blocks | [Link](./fa2_single_head_v2.cu) | 35.74ms | 21.04% | 92.73% | Uncoalesced shared accesses est speedup 86.73%, shared load bank conflicts est speedup 78.20%, L1TEX local store access pattern est speedup 74.97%. Matrix multiplication is primary memory overhead.
| V3 | Matrix multiplication multiplies (A @ B) instead of (A @ B.T) | [Link](./fa2_single_head_v3.cu) | 9.48ms | 79.28% | 79.28% | L1TEX local store access pattern est speedup 55.43%; Memory I/O causing warp stalls. `matrix_block_load_transpose()` seems to have a big memory overhead.
| V4 | Faster matrix multiplication using registers based on https://siboehm.com/articles/22/CUDA-MMM | [Link](./fa2_single_head_v4.cu) | 43.98ms | 53.32% | 53.32% | Why is this slower than V3? Seems to be using local memory not registers.
| V5 | Add padding to matrix load transpose to reduce smem bank store conflicts | [Link](./fa2_single_head_v5.cu) | 9.45ms | 79.61% | 79.61% | Matrix multiplication needs to be improved. li_update and mi_update also have excessive L1 wavefronts.