# Mixed precision (released in 0.6.0)

Status: built; tested on one machine per GPU kind (CUDA: an RTX 4060 Laptop GPU on Windows, bf16 products; Metal: an Apple GPU of the Apple7 family on macOS, f16 products; up to two processes sharing the GPU; shown: one process, two-rank data parallelism and two-rank sharded training (stage 2) each run twice with every per-step weight hash the same, each resumed from a checkpoint in a new launch with every per-step hash the uninterrupted run's, the per-step loss within a tolerance of f32's over 12 steps (set with the measurement), the mixed key apart from f32's in both distributed modes, and with every option off the 0.5.0 results unchanged); multi-GPU, multi-machine and datacenter runs untested. Mixed-precision results are their own kind: never compared byte for byte with, or pooled with, f32 results.

## What it is

- **Products at half precision.** Every matrix product of the forward and backward passes runs at `tensor::MatmulPrecision::Half` (`doc/matmul_precision.md`): its operands rounded to bf16 (CUDA's tensor cores) or f16 (Metal's simdgroup matrices) as they are loaded, accumulated in f32. Everything else (elementwise operations, softmax, reductions, the loss) is f32, as before.
- **f32 master weights and state.** The vars, the gradients and the optimizer's state (AdamW's moments) stay f32, and so does the optimizer's arithmetic. There is no 16-bit copy of the weights to keep in step.
- **A static loss scale** (`nn::LossScale`, 2^k): each micro-step's loss is multiplied by 2^k before its backward pass and the summed gradients by 2^−k after it. A power of two makes both exact in f32 (outside overflow and the subnormal range), so the scale changes only what the 16-bit product operands see: small gradients stay inside f16's range. bf16 has f32's exponent range and needs no scale; f16's smallest normal value is 6.1e-5. When any gradient element on any rank is not finite, every rank skips the optimizer step (the ranks agree by an all-gather) and the step reports it (`MixedStep::stepped`). The scale is static: no schedule changes it, so a checkpoint holds no extra state, and a run resumes exactly. A dynamic scale (raised and lowered from the overflow history) is not built; it would be a state for the checkpoint.

## Using it

```rust
use noetic::nn::dist::{mixed_data_parallel_step, GroupOptions, ProcessGroup, Reduce};
use noetic::nn::LossScale;
use noetic::tensor::{set_matmul_precision, MatmulPrecision};

set_matmul_precision(MatmulPrecision::Half);       // before the group is made: its key names the product
let mut group = ProcessGroup::from_env(GroupOptions::new("my run").with_device(device))?;
let scale = LossScale::power_of_two(8)?;           // f16 (Metal); LossScale::none() for bf16
for step in 0..steps {
    let s = mixed_data_parallel_step(&mut group, &mut vars, &mut opt, k, Reduce::Mean, 1.0, scale, |l, i| loss_of(l, step, i))?;
    if !s.stepped { /* every rank skipped this step */ }
}
```

- `mixed_data_parallel_step`, `mixed_data_parallel_grads` (for a caller that clips first) and `mixed_sharded_step` (with `ShardedDataParallel`, any stage and partition), in the deterministic mode or the fast mode (`DistMode::Fast`), whichever the group is in.
- `mixed_record(group, scale)`: the group's record (its key names the product), `matmul=half` and the loss scale.
- `examples/ddp_gate.rs --mixed [--loss-scale K] [--zero S:P] [--fast]`.

## Keys and refusals

- **The key.** A group made while the thread's precision is `Half` records it, and its platform key's device part ends with the product's part (`-gemm-bf16-mma.m16n8k16-f32acc-tile64x64x32-kernels-<hex>` on CUDA, `-gemm-f16-simdgroup8x8-f32acc-tile64x64x32-kernels-<hex>` on Metal); in the deterministic mode the key is `mode=deterministic order=flat-rank …` with that part, in the fast mode `mode=fast order=ring world=… bucket=… …` with it. A mixed key never equals an f32 key.
- **Refused:** a mixed step on a group made at f32 (its key would not say so); a data-parallel or sharded step taken while the thread's precision differs from the group's (`check_precision`, for every step, mixed or not); a job whose ranks multiply at different precisions (another platform key at the rendezvous, before step 0); a resume of a mixed checkpoint at f32, and the other way round (the checkpoint holds the group key).
- **The loss scale** is part of the run's configuration: put it in the job key (as `ddp_gate` does for a scale other than 2^0) so that a resume under another scale is refused.

## The deterministic mode's identity, as far as shown

Shown on CUDA and Metal (the gate's model, 12 steps): within the mixed key, two-rank data parallelism with one micro-step per rank and two-rank sharded training (stage 2, flat) gave the weights of one process with two micro-steps, bit for bit. Other world sizes, micro-step splits, stages and partitions under mixed precision are untested. The reasons the identity is expected (each micro-step's gradient computed by the same kernels on the same inputs on every rank, `GradSum`'s flat rank order, an exact scale and unscale, the same skip decision everywhere) are not a showing. It is an identity within the mixed key, never a comparison with f32.

## The gates (`doc/gates/e3/`; logs in `doc/gates/e3/logs/`)

The gate's model and data (a decoder of width 32, 2 blocks, MLP 64, 4 heads, context 8, dropout 0.1 keyed by the global micro-step, 2 models on the seed axis, AdamW at lr 0.01, micro-batches of 2 items from the 24-item copy task); every rank on the machine's one GPU; per-step weight checks on every rank.

### Gate 2: run to run and restart, per key (12 steps, M = 2 global micro-steps)

| Configuration | CUDA (bf16) | Metal (f16) |
|---|---|---|
| one process, run twice | every per-step hash the same | every per-step hash the same |
| one process, 7 steps with checkpoints every 3, a new launch resuming from step 6 to 12 | every per-step hash the uninterrupted run's | the same |
| W 2 data-parallel, run twice; resumed as above | the same; the same | the same; the same |
| W 2 sharded (stage 2, flat), run twice; resumed as above (consolidated checkpoints) | the same; the same | the same; the same |
| W 2 data-parallel and sharded against one process | every per-step hash equal | every per-step hash equal |
| final weights (all of the above) | `e7c87fa3ce04369e172c4e0ce66651311286cb22a4899ec105266db8e0efbcc3` | `05801ed7a2c10db982eb9e33cdbb04cea3395fc2db74bfbea2c306746813704c` |
| fast mode, W 2, M 4, run twice | the same per-step hashes (an observation) | the same per-step hashes (an observation: the fast mode makes no such promise) |
| an f32 run resuming the mixed checkpoint | refused (the key) | refused (the key) |
| a job with one rank mixed and one at f32 (the same job key) | refused before step 0 (the library test, ranks as threads) | refused on both ranks before step 0 (two launchers) |

Library tests (`nn::dist::tests::mixed_tests`, ranks as threads on the GPU; CUDA and Metal; logs `win-cuda-tests.log`, `mac-metal-tests.log`): W 2 data-parallel and sharded (stage 2) equal one process for 6 steps; a scaled backward pass that overflows f32 (the loss times 1e30 under a scale of 2^60) skips every step on both ranks, data-parallel and sharded, leaving the vars' initial bits, while a scale of 2^8 steps; a job mixing precisions, and a step at another precision than the group's, are refused; the loss scale's round trip is exact on values across the normal range and the finiteness check finds an infinity or a NaN anywhere.

### Gate 3: the loss against f32 (a tolerance set with the measurement, not an identity)

One process, M = 4, 200 steps; the per-step mean loss over the 4 micro-steps (the masked cross-entropy summed over the 2 models' answer positions).

**The tolerance, set with the measurement:** over the first 12 steps, the mixed run's per-step loss was within 8.04e-3 of f32's on bf16 and within 1.74e-3 on f16 (absolute; the loss starts near 6.6); the tolerances 2e-2 (bf16) and 2e-3 (f16) were set from this measurement, not fixed in advance. Measured:

| | Largest difference, steps 1–12 | Mean loss by window (steps 1–12 / 13–50 / 51–100 / 101–150 / 151–200) | Skipped steps |
|---|---|---|---|
| Metal f32 | – | 4.2007 / 0.6709 / 0.1209 / 0.0799 / 0.0735 | – |
| Metal f16, scale 2^0 | 1.74e-3 (relative 6.6e-4) | 4.2008 / 0.6710 / 0.1058 / 0.0694 / 0.1585 | 0 |
| Metal f16, scale 2^8 | 1.18e-3 (relative 4.4e-4) | 4.2009 / 0.6710 / 0.1174 / 0.0908 / 0.0896 | 0 |
| CUDA f32 | – | 4.2007 / 0.6709 / 0.1209 / 0.0807 / 0.0834 | – |
| CUDA bf16, scale 2^0 | 8.04e-3 (relative 1.35e-3) | 4.1991 / 0.6712 / 0.1024 / 0.1247 / 0.0891 | 0 |
| CUDA bf16, scale 2^8 | the same losses as scale 2^0 | the same | 0 |

The runs follow f32 closely while the loss falls (the window means agree to 1e-3 through step 50); later, each step's loss on this small task moves by a large factor from batch to batch in every precision (f32's own loss rises from 0.016 at step 125 to 0.096 at step 200), and the runs' trajectories, which differ in their last bits from the first step, part (largest single-step difference over 200 steps: 0.71 at f16 with scale 2^0, 0.40 with 2^8). On f16 the run with scale 2^8 stayed nearer f32's late window means than the run without; one seed and one task do not establish a rule. On bf16 the scale changes nothing (the same losses at 2^0 and 2^8): bf16 has f32's exponent range, so rounding a value scaled by a power of two gives the scaled rounding. Whether a run needs a loss scale is a matter for the f16 path only.

### Gate 1: with every option off, nothing changes

- `examples/ddp_gate.rs`'s 26 reference and two-rank results print 0.5.0's final weights line for line on the Mac CPU and on Metal (`ddp_gate_refs.sh`),, and 36 on Windows (CUDA and the CPU; `win_e3.ps1 -Part off`, through `doc/gates/e2/win_gates.ps1`).
- The six single-device configurations and the copy task print the same bytes built from the 0.5.0 release and from this tree, on the Mac CPU and Metal (`single_device_diff.sh`), the Windows CPU and CUDA (`win_e3.ps1 -Part off`).
- The `nn::dist` tests pass on the Mac CPU (61, the runs needing more than 2 processes at once skipped; `mac-cpu.log`); the Metal GPU tests and the mixed tests pass (`mac-metal-tests.log`).
- The changes outside the new steps: `ProcessGroup` records the matmul precision it was made at (F32 unless the thread set Half: the key and the record are unchanged for f32), and `data_parallel_grads` and `ShardedDataParallel::grads` check it (an error only when the thread's precision differs from the group's).

## Untested and not built

- Untested: multi-GPU and multi-machine runs; stages 1 and 3 and the by-var partition under mixed precision (they take the same steps; gated: stage 2, flat); clipping under mixed precision in a gate (built: `mixed_data_parallel_grads` returns the finite flag before the caller clips); long training at scale.
- Not built: a dynamic loss scale; 16-bit weights or activations in memory (activations are f32 tensors; only the products' operands are rounded); bf16 on Metal; a separate learning-rate schedule for skipped steps (a skipped step is not counted by the optimizer).
