# Reduced-precision matrix products on the GPU (released in 0.6.0)

Status: built; tested on one machine per GPU kind (CUDA: an NVIDIA GPU, compute capability 8.9, on Windows; Metal: an Apple GPU of the Apple7 family on macOS; shown: every product within the stated tolerance on fixed shapes, a training run repeated in a new process and resumed from a checkpoint in a new process with every per-step weight hash the same, the key apart from f32's, and with the option off the 0.5.0 results unchanged); multi-GPU, multi-machine and datacenter runs untested. Its results are its own kind: they are never compared byte for byte with, or pooled with, f32 results.

## What it is

`MatmulPrecision::Half` multiplies GPU tensors on the GPU's matrix units: both operands are rounded to a 16-bit float as they are loaded, and the products are accumulated in f32. The output is f32.

| Backend | Operands | Instruction | Accumulator | Tile, k order |
|---|---|---|---|---|
| CUDA (compute capability 8.0 or later) | bf16, round to nearest, ties to even (NaN kept) | tensor cores, `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | f32 | 64×64 output tiles, 128 threads (four warps, 32×32 each); k in steps of 32, in order |
| Metal | f16, round to nearest even; beyond ±65504 an operand becomes ±∞ | simdgroup 8×8 matrices (`simdgroup_multiply_accumulate`) | f32 | 64×64 output tiles, 128 threads (four simdgroups, 32×32 each); k in steps of 32, in order |

- **One configuration per backend.** One kernel, whatever the shape; no timing, no autotuning, no split over k, no atomics. Each output is one chain of matrix instructions over k in order, so the same product gives the same bits on the same GPU, driver and build (checked run to run, below). How the loads are done (four values at a time for interior tiles of contiguous or transposed operands, one at a time at edges) moves values; it does not change what is added.
- **Its own source.** Each kernel is a source of its own (`tensor::cuda::half_gemm`, `tensor::metal::half_gemm`), compiled on the first reduced-precision product only. The f32 kernels' sources, their hashes and the device keys are unchanged while the option is off; cuBLAS is not used.
- **The operands are f32 tensors.** The rounding happens in the kernel; bf16- or f16-typed tensors are not taken as operands (they are cast to f32 first, as for the f32 product).
- **CPU tensors** multiply in f32 whatever the thread's setting; `matmul_with(…, Half)` on a CPU tensor is refused.

## The option and its key

- `tensor::set_matmul_precision(MatmulPrecision::Half)` sets the precision of `Tensor::matmul` on GPU tensors for the calling thread (it returns the previous setting; `matmul_precision()` reads it). `F32` is the default.
- `Tensor::matmul_with(rhs, precision)` names the precision of one product.
- A product's backward pass runs its two products (the gradients of both operands) at the forward product's precision.
- **The key.** While the calling thread's precision is `Half`, `tensor::gpu_platform_key(device)` ends with the product's part, `-gemm-<operands>-<instruction>-f32acc-tile64x64x32-kernels-<16 hex digits of the kernel source's sha256>` (`tensor::half_gemm_key(device)`):
  - CUDA: `…-driver-release-<release>-gemm-bf16-mma.m16n8k16-f32acc-tile64x64x32-kernels-<hex>`;
  - Metal: `…-macos-Version_27.0.1__Build_26A434_-gemm-f16-simdgroup8x8-f32acc-tile64x64x32-kernels-<hex>`.
  With `F32` the key is 0.5.0's. A process group made while the precision is `Half` carries the part in its key (`nn::dist::device_key` reads `gpu_platform_key`), so a checkpoint taken at one precision is refused at the other (shown below), and a job whose ranks use different precisions is refused at the rendezvous (it follows from the key; not run). Set the precision before the run's keys are taken.

## The gates (`doc/gates/c1/`; logs in `doc/gates/c1/logs/`)

### The error on fixed shapes, as a tolerance

`tensor::tests::gpu::{cuda,metal}::half_gemm` (logs `win-cuda-half.log`, `mac-metal-half.log`). Operands uniform in [−2, 2). For each output, the error is divided by Σₖ|aᵢₖ·bₖⱼ| of the operands; the table gives the largest over the outputs, against two f64 references: the exact product of the rounded operands (what the kernel computes, up to the f32 accumulation), and the exact product of the f32 operands.

**The stated tolerance** (asserted by the test): against the rounded operands, at most 2⁻²¹·k; against the f32 operands, at most 2u + 2⁻²¹·k, with u the 16-bit type's unit roundoff (bf16 2⁻⁸, f16 2⁻¹¹).

| Shape (batch, m×k · k×n) | CUDA bf16: rounded operands | CUDA bf16: f32 operands | Metal f16: rounded operands | Metal f16: f32 operands |
|---|---|---|---|---|
| 1, 64×64·64×64 | 8.5e-8 | 1.20e-3 | 1.5e-7 | 1.8e-4 |
| 1, 256×256·256×256 | 1.9e-7 | 7.4e-4 | 1.9e-7 | 9.6e-5 |
| 2, 130×300·300×257 (ragged edges) | 1.8e-7 | 7.7e-4 | 2.4e-7 | 8.4e-5 |
| 1, 7×5·5×3 | 3.4e-8 | 3.45e-3 | 7.5e-8 | 4.6e-4 |
| 1, 64×2048·2048×64 (long k) | 3.6e-7 | 2.8e-4 | 2.2e-7 | 2.5e-5 |
| 3, 100×96 · 1, 96×70 (broadcast batch) | 9.5e-8 | 1.20e-3 | 1.8e-7 | 1.5e-4 |
| 2, Aᵀ·Bᵀ of views (k 150) | 1.6e-7 | 9.5e-4 | 2.4e-7 | 1.2e-4 |
| 1, 512×512·512×512 | 2.2e-7 | 6.5e-4 | 2.8e-7 | 7.5e-5 |

The error against the f32 operands is the operands' rounding (up to 2u, bf16's 2⁻⁸ ≈ 3.9e-3 and f16's 2⁻¹¹ ≈ 4.9e-4); the accumulation adds at most 3.6e-7. The f32 path's own error on the same shapes is at most 2.5e-7.

### Run to run and restart (kind (ii))

`examples/half_gemm_gate.rs` (the gate's decoder: width 32, 2 blocks, MLP 64, 2 models on the seed axis, AdamW at lr 0.01, a copy task of 2 fresh items per step), `doc/gates/c1/gate.sh` (Metal) and `doc/gates/c1/win_c1.ps1 -Part gate` (CUDA):

| | CUDA, an NVIDIA GPU | Metal, Apple GPU |
|---|---|---|
| 200 steps, run twice, each its own process | every per-step weight hash the same; final `d7e6daf3c00737ca84af587f628c0a76f13ac518afe3cfcdbf6ae1e4c6e1e674` | every per-step weight hash the same; final `d206005f238b8d76c49af6da694a92c7b6ae8c53b45a8f1c4f149380c3ff63eb` |
| 107 steps with checkpoints every 50, then a new process resuming from step 100 to 200 | every per-step hash the uninterrupted run's | the same |
| an f32 run resuming the half checkpoint | refused (the key) | refused (the key) |
| a larger decoder (width 256, 4 blocks, MLP 1024, 32 items per step: full 64×64 tiles), 20 steps, run twice | every per-step hash the same; final `31ea7099354a2ad1df17830b4401266470fcc533e74cb8ec2502625b130d5c31` | the same; final `9eef90f8a0d2a2281f07ad95e12cf496a4938ed4c3ddcb2f6193f92fe30c10e5` |

The library test `body_half_matmul_is_run_to_run_identical` repeats eight products and two backward gradients in one process with the same bits on both GPUs. These identities hold on one machine, build and key; they are the property a kind (ii) option must show before use, not a comparison with f32.

### Training against f32 (a tolerance, not an identity)

The same 200 steps at f32 (each step's loss is the masked cross-entropy summed over the 2 models' answer positions):

| | Largest loss difference, steps 1–10 | Mean loss, steps 1–10: f32 / half | Mean loss, steps 191–200: f32 / half | Largest difference, all 200 steps |
|---|---|---|---|---|
| CUDA (bf16) | 1.34e-2 | 6.5381 / 6.5374 | 0.0154 / 0.0158 | 0.75 (step 131) |
| Metal (f16) | 1.68e-3 | 6.5381 / 6.5386 | 0.0154 / 0.0165 | 0.26 (step 155) |

Both train to the same loss level; over many steps the trajectories drift apart as the rounding differences compound (as any change of arithmetic does), so the step-by-step difference late in training is a trajectory's, not a product's.

### Wall time (a measurement on these machines only)

The baseline is the engine's own deterministic f32 kernels (CUDA: compiled with `--fmad=false`, one explicit fmaf per k in order; Metal: safe math mode), not cuBLAS or any other vendor library; the ratios below are not a comparison with vendor libraries.

The library's throughput test (`half_matmul_throughput`, ignored by default; median of 21 runs, 7 for the largest; resident operands, synchronised):

| Product | CUDA f32 | CUDA bf16 | Metal f32 | Metal f16 |
|---|---|---|---|---|
| 1024³ | 0.700 ms (3.07 TFLOP/s) | 0.226 ms (9.52) | 2.498 ms (0.86) | 1.701 ms (1.26) |
| 2048³ | 5.190 ms (3.31) | 1.388 ms (12.38) | 17.473 ms (0.98) | 11.712 ms (1.47) |
| 4096³ | 38.503 ms (3.57) | 12.044 ms (11.41) | 134.411 ms (1.02) | 102.407 ms (1.34) |
| 512×512·512×2048 | 0.358 ms (3.00) | 0.129 ms (8.32) | 1.493 ms (0.72) | 0.943 ms (1.14) |
| 4096×768·768×3072 | 5.300 ms (3.65) | 1.350 ms (14.31) | 18.954 ms (1.02) | 12.222 ms (1.58) |
| A·Bᵀ, 512×2048·2048×512 | 0.350 ms (3.07) | 0.123 ms (8.73) | 2.488 ms (0.43) | 1.421 ms (0.76) |
| Aᵀ·B, 512×512·512×2048 | 0.292 ms (3.68) | 0.141 ms (7.60) | 1.486 ms (0.72) | 1.004 ms (1.07) |
| 32 × 256×64·64×256 | 0.134 ms (2.00) | 0.068 ms (3.97) | 0.536 ms (0.50) | 0.434 ms (0.62) |

Training (`half_gemm_gate`: width 512, 4 blocks, MLP 2048, 16,802,304 parameters per model, 2 models, 64 items per step, 10 steps without the per-step hash; the loop's wall time, median of 3 runs in rotation): CUDA f32 0.485 s, bf16 0.403 s; Metal f32 1.502 s, f16 1.360 s. At this size the products are a part of the step only.

Hardware and keys: CUDA on an NVIDIA GPU (NVRTC 12.6) on the Windows x86_64 host, every process on 2 logical CPUs at below-normal priority, key `cuda-<the GPU's name>-sm89-driver<version>-nvrtc12.6-nofastmath-fmadoff-kernels-<hex>-driver-release-<release>` (f32) and that key with `-gemm-bf16-mma.m16n8k16-f32acc-tile64x64x32-kernels-<hex>` (bf16); Metal on an Apple GPU (Apple7 family), macOS 27.0.1, every process at background priority (`taskpolicy -b`), on a machine shared with other work, key `metal-<the GPU's name>-mathmode-safe-kernels-<hex>-family-apple7-metal4-mac2-arch-<architecture>-macos-Version_27.0.1__Build_26A434_` (f32) and that key with `-gemm-f16-simdgroup8x8-f32acc-tile64x64x32-kernels-<hex>` (f16).

### With the option off, nothing changes

- `examples/ddp_gate.rs`'s references print 0.5.0's final weights line for line: 26 lines on the Mac CPU and 26 on Metal (`doc/gates/c1/ddp_gate_refs.sh`), 36 on Windows (CUDA and the CPU; `win_c1.ps1 -Part off`, through `doc/gates/e2/win_gates.ps1`).
- The six single-device configurations and the copy task print the same bytes built from the 0.5.0 release and from this change, on the Mac CPU, Metal, the Windows CPU and CUDA (`single_device_diff.sh`; `win_c1.ps1 -Part off`).
- The Metal GPU tests (the f32 kernels bit-exact against the CPU reference backend, keys, the backward pass) pass (`mac-metal-gpu-tests.log`), and the library's tests pass.
- The changes outside the two new kernels: `Tensor::matmul` passes the thread's precision on (F32 by default: the same call as before), `matmul_with`, the GPU backends' `matmul_half` and `half_gemm_key`, the second module (CUDA) and library (Metal) loaded on first use, and `gpu_platform_key`'s part. The f32 matmul code is unchanged.

The option-off checks on Windows ran on the first commit of this change (the later commits change only the two reduced-precision kernels' loads, their test and the gate example); the Mac checks ran on the same commit's f32 code.

## Untested and not built

- Untested: other GPUs (CUDA's kernel needs compute capability 8.0; Metal's was run on the Apple7 family only), multi-GPU and multi-machine runs, long training at scale. Inside data-parallel and sharded jobs the product was run by mixed precision (`doc/mixed_precision.md`: two ranks, data-parallel and sharded stage 2); other configurations there are untested.
- Not built: bf16 on Metal; f16 or tf32 on CUDA; 16-bit tensors as operands without a cast; tiles chosen by shape; split over k; cuBLAS. (Mixed precision with f32 master weights and a loss scale is built on this product in the same release: `doc/mixed_precision.md`.)
