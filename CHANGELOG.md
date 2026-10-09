# Changes

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
