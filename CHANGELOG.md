# Changes

## Unreleased

**Sharded data parallelism** (`nn::dist::ShardedDataParallel`, deterministic mode only): built; tested on one machine per device kind (the CPU reference backend on macOS aarch64 at world sizes 1 and 2 and on Windows x86_64 at 1, 2 and 4; CUDA with up to four processes sharing a single CUDA GPU on Windows; Metal with two processes sharing one Apple GPU on macOS; shown: byte-identical weights and per-step hashes to data parallelism at every stage and partition, with absent gradients, an idle weight under AdamW's decay and clipping, consolidated checkpoints byte-identical to data parallelism's, and a killed run resumed from a sharded checkpoint equal to the uninterrupted run); multi-GPU, multi-machine and datacenter runs untested. See "Sharded data parallelism" in `doc/distributed.md`.

- **Stages.** The optimizer's state (stage 1), also the summed gradients (stage 2), and also the vars between steps (stage 3) are split among the ranks, in the shape of ZeRO's stages and FSDP's full sharding; `ShardOptions` (non-exhaustive, with a builder) picks the stage and the partition: flat pieces of every var (FSDP's flat parameter, per var) or whole vars per rank (in the shape of torch's `ZeroRedundancyOptimizer`). Opt-in: nothing else changes when it is not used.
- **The same bits as data parallelism.** The reduce-scatter is the all-reduce's chain in flat (rank, micro-step) order, with only its last step cut into pieces; the optimizers update each element on its own; gathers move bits. At every stage and partition the weights and per-step hashes equal `data_parallel_step`'s at the same world size and micro-steps, with absent gradients, an idle weight under AdamW's decay, and clipping by the global norm per seed (`ShardGrads::clip_grad_norm_per_seed`, whose norms are `clip_grad_norm_per_seed`'s bit for bit). The group key is unchanged; the run's record names the sharding.
- **Checkpoints.** Consolidated checkpoints are data parallelism's checkpoints byte for byte (either kind of run resumes the other's); sharded checkpoints have every rank write and verify its own pieces under a meta file of sha256s, round-trip byte for byte, and resume a killed run to the uninterrupted run's weights; another world size, partition, key or optimizer is refused.
- **Memory.** `ShardedDataParallel::memory` reports the bytes a rank holds; on the gate's model each rank's optimizer state is 1/W of data parallelism's with flat pieces.
- **`examples/ddp_gate.rs`**: the gates' training as an example binary the launcher starts, each rank on the device its rank on the node gives (one GPU per rank, or several sharing one); alone, it is the one-device reference with accumulation. It prints per-step weight hashes, the final weights' hash and the run record, and takes checkpoints, resume and node copies. Built; tested on one machine per device kind (ranks sharing the machine's one device: the CPU on macOS and Windows, one CUDA GPU, one Apple GPU); one GPU per rank is untested (no multi-GPU machine available).
- **Not built:** per-layer gathering, prefetching and overlap, offloading, mixed precision, sharding outside the flat order.
- **Internal:** `ProcessGroup` gains a reduce-scatter beside its all-reduce (one more message kind on the wire); `GradSum` and the reduce-scatter share one scaling routine. Data-parallel results are unchanged (the 0.4.0 gates give 0.4.0's hashes).

## 0.4.0

**Data-parallel ranks on a GPU** (`nn::dist`, deterministic mode only): built; tested on one machine per GPU kind (CUDA: Windows x86_64, a single CUDA GPU shared by up to four processes; Metal: macOS aarch64, one Apple GPU shared by two processes; shown: the same training twice byte-identical on the GPU, and byte-identical weights and per-step hashes for 2 and 4 ranks against 1 GPU with accumulation, with absent gradients (−0 and +0), an idle weight under AdamW's decay, gradient clipping, and a killed and resumed run under AdamW and under SGD with momentum); multi-GPU, multi-machine and datacenter runs untested. Windows ran every gate with no skips; the Mac, shared with a long benchmark, skipped the runs needing 4 processes at once. See "GPU ranks" in `doc/distributed.md`.

- **One device per rank.** `DistEnv::device(kind, ranks_per_device)` gives the rank's device from `LOCAL_RANK` (torchrun's `cuda:LOCAL_RANK`; several ranks may share a GPU); `GroupOptions::device` is the rank's device, and the data-parallel step refuses vars on another device. Gradients are copied to the host and summed by `GradSum` in the flat rank order, as in 0.3.0; every rank takes the same optimizer step on its device. No overlap or bucketing yet.
- **`GroupOptions`** gains the field `device` and is now `#[non_exhaustive]`, with `GroupOptions::new` and `with_*` builder methods: struct literals no longer compile outside the crate (breaking), and later fields will not break callers again. Its fields stay public to read and set.
- **Keys.** On a GPU the platform key gains `device=` and the GPU's platform key (`tensor::gpu_platform_key`): on CUDA the GPU model and compute capability, the driver's release and the CUDA version it supports, NVRTC's version and the kernel source's hash; on Metal the GPU, its families and architecture, the OS build and the kernel source's hash. CPU and GPU results never share a key and are never compared byte for byte; a job mixing devices is refused at the rendezvous. The CPU reference backend's key is unchanged.
- **NVML** (the driver's management library) gives the driver's release. It is loaded at run time, not linked: the `cuda` feature builds, and a process trains on one GPU, where NVML is missing; there `device_key` is an error, and a GPU key is never made without the driver's release.
- **The fast CPU backend is refused** as a device of the deterministic mode (`device_key(Cpu(Fast))` is an error); a fast mode returns later only under its own key.
- **Resume** refuses a checkpoint taken under another group key (another platform or device), besides another world size or job key.
- **SGD with momentum is checkpointed.** `Checkpoint::save` and `restore` take any `CheckpointOptimizer` (Adam or SGD); an SGD checkpoint holds the momentum buffers, and a resume into another kind of optimizer than the checkpoint's is refused, so no run resumes without its optimizer state. Adam checkpoints are written byte for byte as in 0.3.0. `Checkpoint` holds `optimizer_state` in place of `adam_m` and `adam_v`, and `CheckpointMeta` gains `optimizer` (breaking).
- **`data_parallel_grads`** returns a step's summed gradients, the same on every rank, for callers that clip them before their optimizer steps.
- **CUDA backend:** runs on `Cuda(i)`, one device per process (fixed by the first use; `Cuda(0)` by default); a device the machine lacks is refused. `Cuda(i)` for i > 0 is untested, and results that must be exact should not rely on it until a two-GPU gate passes.
- **Single-device behaviour is unchanged.** The six training configurations give byte-identical final weights and logits to 0.3.0 (0.3.0 built and run now as the reference) on the CPU reference backend (macOS aarch64, Windows x86_64), on CUDA and on Metal; the 0.3.0 CPU gates give 0.3.0's hashes. No kernel source changes.
- **`tensor::cuda::kernel_ptx_and_cubin(cc)`** (diagnostics): NVRTC's PTX for the kernel source, byte for byte the PTX the backend loads, and the CUBIN NVRTC makes for the real architecture (`nvrtcGetCUBIN`), so a GPU run can record its PTX and machine code (NVRTC's machine code, which need not be the driver's compilation of the PTX). Tested on an RTX 4060 Laptop GPU.
- **Tests:** the kernel sources hold no atomic operations; `DIST_TEST_MAX_PROCS`, a development switch, lets a shared machine skip the gate runs needing more processes at once (each skip is printed).

**Tested** (measured on 0.3.0's code, before this release's changes):

- Built and tested on Linux x86_64 (CPU) and on NVIDIA A100-SXM4-80GB and L4 (NVIDIA driver 595.91.07, CUDA driver API 13.2; NVRTC 12.8, cuBLAS 12.8.4); determinism gates passed: run-to-run identical within each GPU model and key (sm80, sm89).
- On one 200-step training probe, the CUDA trace was byte-identical on A100, L4 and an RTX 4060 Laptop GPU (two drivers, two NVRTC/cuBLAS versions). This is a measured instance, not a guarantee: other workloads are untested, and results remain keyed per GPU model and driver.

## 0.3.0

**Data-parallel training on the CPU** (`nn::dist`), with an API in the shape of torch.distributed, DDP and torchrun: a process group from torchrun's environment variables, rank-ordered collectives over TCP (standard library only, no new dependency), a data-parallel step, a sharded sampler, coordinated checkpoints and a launcher (`examples/launch.rs`). See `doc/distributed.md`.

- **Determinism.** Deterministic data-parallel training on the CPU: W processes with k accumulation micro-steps each give the weights of one process with W·k, bit for bit; shown at world sizes 1, 2 and 4 on one machine (localhost TCP). Multi-machine and GPU runs are untested.
  - Where the identity holds: weights are byte-identical across world sizes only at equal totals of global micro-steps (W·k), in the flat rank order, on one platform key.
  - The platform key includes a fingerprint of the OS maths library (libm) over fixed inputs; it is checked at the rendezvous, and a job whose ranks differ is refused before step 0.
  - A per-step weight check (`check_in_sync`) is the backstop for anything the fingerprint does not see.
  - Resume at a different world size is refused.
- **Security.** The rendezvous handshake is authenticated (HMAC-SHA256 over a per-job secret); collective traffic after it is plain TCP, neither authenticated nor encrypted: run on private networks only.
  - A multi-node job needs a per-job secret: the launch command refuses a multi-node launch without `--secret-file` (or an explicit `--no-auth`, for an isolated development network only).
  - The secret never appears in the `Debug` or `Display` forms, run records, logs or checkpoints.
- **Checkpoints:** every rank verifies the bytes rank 0 writes (sha256); resume is coordinated and refused on a key mismatch.
- **API.**
  - `NnError` is now `#[non_exhaustive]` and gains `Dist`: a breaking change for exhaustive matches, made in this pre-1.0 minor release.
  - `GroupOptions` gains public fields `platform` and `secret`.
  - Every other 0.2.0 public item and every 0.2.0 test is kept.
- **Single-device behaviour is unchanged.** Six training configurations (a decoder, a mixture-of-experts decoder and a decoder with dropout, each with AdamW and with SGD with momentum) give byte-identical final weights and logits to 0.2.0.
- **GPU sources.** No CUDA or Metal source changes in this release. CUDA was checked by a compile-only build (`cargo build --features cuda`, Windows x86_64, CUDA 12.6); nothing was run on a GPU.
- **Tested:** CPU, localhost, Mac (macOS aarch64). **Not tested:** multi-node over a real network, GPU, Linux CUDA (apart from the compile check above, which was on Windows).

## 0.2.0

- Comments in the CUDA and Metal kernel sources are reworded (no code change); the kernel-source hashes, and with them the device keys, change.
- The gated MLP takes an optional caller-supplied extension (`MlpExtension`, `MlpTransform`) that may add vars and transform the gate's pre-activation and the hidden activations.
- `tensor::cuda_kernel_source()`: the CUDA kernel source text, available without the cuda feature.
- `tensor::api`: a typed facade in burn 0.21's shape.
- Every 0.1.0 test and public item is kept.
- Breaking: the library is imported as `noetic` (`use noetic::tensor::…`), no longer as `noetic_tensor`.
- Breaking for exhaustive struct literals and matches: `DecoderConfig`, `BlockConfig` and `GatedMlpConfig` gain an extension field (`None` by default; `..DecoderConfig::new(…)` is unaffected); `MlpKind` gains `SwiGlu`; the error type is `NnError`, with `Error` kept as an alias, and gains the variant `ExtensionNeedsGatedMlp`.
- `rng` is public.
- Licence: MIT OR Apache-2.0 (`LICENSE-MIT`, `LICENSE-APACHE`).

## 0.1.0

The first release.
