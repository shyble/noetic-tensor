# Changes

## Unreleased

**Data-parallel ranks on a GPU** (`nn::dist`, deterministic mode only): built; tested on one machine per GPU kind (CUDA: Windows x86_64, one RTX 4060 Laptop GPU, two processes sharing it; Metal: macOS aarch64, one Apple GPU, two processes sharing it; shown: byte-identical weights and per-step hashes for 2 ranks against 1 GPU with accumulation at 2 and 4 global micro-steps, a killed and resumed run equal to the uninterrupted run, and the same training twice byte-identical on the GPU); multi-GPU, multi-machine and datacenter runs untested (no multi-GPU machine available yet). See "GPU ranks" in `doc/distributed.md`.

- **One device per rank.** `DistEnv::device(kind, ranks_per_device)` gives the rank's device from `LOCAL_RANK` (torchrun's `cuda:LOCAL_RANK`; several ranks may share a GPU); `GroupOptions::device` is the rank's device, and `data_parallel_step` refuses vars on another device.
- **The same ordered sum.** Gradients are copied from the device to the host as f32 and summed by `GradSum` in the flat rank order, as in 0.3.0; every rank takes the same optimizer step on its device. No overlap or bucketing yet.
- **Keys.** On a GPU the platform key gains `device=` and the GPU's platform key (`tensor::gpu_platform_key`): on CUDA the GPU model and compute capability, the driver's CUDA version and release (NVML), NVRTC's version and the kernel source's hash; on Metal the GPU, its families and architecture, the OS build and the kernel source's hash. CPU and GPU results never share a key and are never compared byte for byte; a job mixing devices is refused at the rendezvous. The CPU reference backend's key is unchanged.
- **Resume** refuses a checkpoint taken under another group key (another platform or device), besides another world size or job key.
- **CUDA backend:** runs on `Cuda(i)`, one device per process (fixed by the first use; `Cuda(0)` by default); a device the machine lacks is refused. Only `Cuda(0)` was available to test. The `cuda` feature now links NVML (part of the driver) for the driver's release.
- **API:** `GroupOptions` gains the public field `device` (breaking for struct literals without `..Default::default()`); new: `DeviceKind`, `DistEnv::device`, `device_key`, `ProcessGroup::device`, `ProcessGroup::platform`, `tensor::gpu_platform_key`. `tensor::gpu_key` is unchanged.
- **Single-device behaviour is unchanged.** The six training configurations give byte-identical final weights and logits to 0.3.0 on the CPU reference backend (macOS aarch64, Windows x86_64), on CUDA and on Metal; the 0.3.0 CPU gates give 0.3.0's hashes (all of them on Windows; on the Mac, which was shared, those that run in at most 2 processes at once). No kernel source changes.
- **Tests:** a test that the kernel sources hold no atomic operations; `DIST_TEST_MAX_PROCS` lets a shared machine skip the gate runs that need more processes at once (each skip is printed).

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
