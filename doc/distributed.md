# Distributed training: deterministic data parallelism (first step)

Released in 0.3.0 (9 October 2026) for the CPU. Unreleased: ranks on a GPU, one device per rank (see "GPU ranks" below); built; tested on one machine per GPU kind (CUDA: two processes sharing one RTX 4060 Laptop GPU on Windows; Metal: two processes sharing one Apple GPU on macOS; byte-identical weights to one GPU with accumulation, and a killed and resumed run equal to the uninterrupted one); multi-GPU, multi-machine and datacenter runs untested (no multi-GPU machine available yet).

## What it adds

`noetic::nn::dist`: data-parallel training across processes (tested on one machine; multi-machine runs are untested), with an API in the shape of torch.distributed, DDP and torchrun. Standard library only (`std::net`, `std::process`); no new dependency.

| Item | What it does |
|---|---|
| `DistEnv` | The process's place in the job, from torchrun's variables: `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE`, `GROUP_RANK`. Without `RANK` and `WORLD_SIZE`, distribution is off. |
| `ProcessGroup` | The rendezvous, a full mesh of TCP connections, and the collectives: `barrier`, `broadcast`, `all_gather`, `all_reduce_f32`, `all_reduce_grads`. `ProcessGroup::single()` (and `from_env` with distribution off) opens no socket; its collectives are the identity. |
| `GradSum`, `Reduce` | The one ordered gradient sum, used by local accumulation and by the all-reduce. |
| `data_parallel_step` | One optimizer step: this rank's micro-steps, the all-reduce, the same optimizer step on every rank. With the single-process group it is gradient accumulation. |
| `check_in_sync`, `state_hash` | Every rank holds the same vars, bit for bit (all-gather of a sha256 of the saved vars). |
| `Shard`, `ShardSpec` | The distributed sampler: each rank's micro-batches, from (seed, rank, world size) only; a sha256 per shard. |
| `Checkpoint` | Coordinated checkpoints: vars, Adam's moments and step count, every rank's loader cursor. |
| `LaunchConfig`, `spawn`, `run`, `Job` | A torchrun-style launcher: N processes per node, rendezvous variables and the job's secret set, a log per rank; the first failure stops the job. |
| `JobSecret`, `platform_key` | The per-job secret of the authenticated handshake; the platform key every rank must share. |
| `DeviceKind`, `DistEnv::device`, `GroupOptions::device`, `device_key` | Unreleased. The rank's device from its rank on the node (torchrun's `cuda:LOCAL_RANK`), the group's device, and the device's part of the platform key (`device=…`, none for the CPU reference backend). |
| `examples/launch.rs` | The launcher as a command: `launch --nproc-per-node N [--nnodes M --node-rank I --master-addr A --master-port P] [--log-dir D] [--secret-file F | --no-auth] -- <command> [args]`. |
| `examples/ddp_copy_task.rs` | The copy task trained data-parallel; the same weights at 1, 2 and 4 processes. |

A training loop:

```rust
use noetic::nn::dist::{check_in_sync, data_parallel_step, GroupOptions, ProcessGroup, Reduce, Shard, ShardSpec};

let mut group = ProcessGroup::from_env(GroupOptions { job_key: "my run".into(), ..Default::default() })?;
let shard = Shard::new(ShardSpec { seed: 7, items: n, micro_batch: 8, micro_steps: 4 }, group.rank(), group.world_size())?;
for step in 0..steps {
    data_parallel_step(&mut group, &mut vars, &mut opt, shard.micro_steps_per_rank(), Reduce::Mean, 1.0, |lifted, i| {
        let items = shard.batch(step, i);                       // this rank's micro-batch i
        let train_step = step * 4 + shard.global_micro(i) as u64; // dropout keyed by the global micro-step
        loss_of(lifted, &items, train_step)                      // the scalar to differentiate
    })?;
}
check_in_sync(&mut group, &vars)?;
```

## Determinism rules (deterministic mode, the only mode so far)

1. **One order for the sum.** A step's gradient is the left fold over its global micro-steps, `((g₀ + g₁) + g₂) + …`, elementwise in f32. Rank r's i-th of k micro-steps is global micro-step `r·k + i`. The fold starts from the first micro-step's gradient (never from an extra zero, which would turn −0 into +0). A var without a gradient in a micro-step (it took no part: a branch not taken, a var unused by that batch) counts as zeros, so the sum is bit for bit the one a single process gets by putting zeros in place of the absent gradients and adding them (`x + 0` included, which turns −0 into +0). A var no micro-step touched stays `None`, and the optimizer skips it, as it does today and as torch does for a parameter unused on every rank; AdamW's decoupled decay is skipped with it, exactly as in one process.
2. **One routine.** `GradSum` is the only place gradients are added, for local accumulation and for the all-reduce. `Reduce::Mean` divides by the number of global micro-steps as `Tensor::div_scalar` does, and not at all when there is one (so a single micro-step passes through bit for bit).
3. **Rank order across the network.** The all-reduce passes the running sum along the ranks: rank 0 sends its sum to rank 1, which folds its own micro-steps in, in order, and sends it on; the last rank sends the total to every other rank once it has all of it. Large sums travel in chunks so the ranks work at the same time; chunking changes the pipelining, never the order of the additions. A rank after the first therefore keeps its micro-steps apart until the running sum reaches it (`ProcessGroup::grad_sum` gives that kind of sum; pre-adding them is refused).
4. **Identical optimizer steps.** Every rank applies the same optimizer to the same summed gradient, so the vars stay identical without being sent. `check_in_sync` verifies it, and every coordinated checkpoint verifies it again.
5. **No timing in the result.** No compression, no quantised or asynchronous reductions, no arrival-order sums. Every message carries its collective's operation and sequence number; ranks that call different collectives fail at once.
6. **The key and pooling.** `ProcessGroup::key()` is the reproducibility key, `mode=deterministic order=flat-rank backend=tcp platform=<platform key>`; on a GPU the platform key ends with the device's part, `device=<GPU platform key>` (see "GPU ranks"), so CPU and GPU results, and two GPU models or drivers, never share a key. The world size is not part of it: by rules 1–3 the weights depend on the number of global micro-steps, not on how they are split among ranks (the gate below shows it), so runs at different world sizes may be pooled on the same platform key. The world size stays in every run's record (`ProcessGroup::record()`, and the checkpoint's meta). Before pooling on a new (platform, device) pair, a short run with per-step weight-hash checks at world sizes 1, 2 and 4 must show the identity there; a GPU needs its own showing. A resume still keeps the checkpoint's world size and job key (it refuses others): pooling compares finished runs, it does not move a run between world sizes. The refusal is lifted only after a gate shows a resume at another world size byte-identical to the uninterrupted run.
7. **Order variants.** The flat order is the default and the only one built. A per-rank-first order (each rank pre-adds its micro-steps, then the ranks' sums are added in rank order) would save the deferred micro-steps' memory but makes the bits depend on the world size; if it is ever added, it is an opt-in whose key includes the world size, and it is never pooled across world sizes.
8. **Randomness inside a micro-step** (dropout) must be keyed by the global micro-step, not by the rank (the gate's model uses dropout 0.1 this way).
9. **Data.** Item p of the global stream is `perm_e[p mod n]` with e = p div n, where `perm_e` is a Fisher–Yates permutation from `rng::derive(seed, "shard epoch e")` (u64 draws, platform-independent). Global micro-batch j of step s starts at item `(s·M + j)·b`. Rank r takes micro-batches `r·k .. (r+1)·k`, so the ranks' micro-batches in rank order are the single process's. The loader cursor is the step number. `Shard::hash` is a sha256 of the shard's key (seed, items, micro-batch, micro-steps, rank, world size) and every item it takes; the training ranks all-gather their shard hashes and check every rank's against their own computation.
10. **A fast mode** (backend-native order, overlap, NCCL's reductions) is not built. It will carry its own key and never be mixed with this mode.

## Rendezvous and failure

- Rank 0 listens on `MASTER_ADDR:MASTER_PORT`. Every other rank connects, answers rank 0's challenge (below), and sends its rank, world size, job key, platform key and the address of its data listener (on the interface it reached rank 0 through). Rank 0 checks that the ranks are distinct and agree on the world size, the job key and the platform key, then sends the address table, or the reason for refusing the job, to every rank. Ranks connect to every lower rank, accept every higher one, and a barrier ends the rendezvous.
- A lost peer is an error on the next message (end of stream), not a hang; a silent peer fails after `GroupOptions::timeout` (600 s by default).
- The launcher stops the whole job when any rank fails or is killed, and names the rank.
- **The authenticated handshake.** The launcher gives every rank the job's secret (`JobSecret`, at least 32 bytes, in `DIST_JOB_SECRET`; a single-node launch makes a fresh one, a multi-node job passes the same `--secret-file` on every node). At the rendezvous and on every link of the mesh each side sends a fresh 32-byte random challenge and the other answers with HMAC-SHA256(secret, label · challenge · its message), computed in-house on the existing sha256 (checked against RFC 4231's vectors). A rank that cannot answer, or answers for another secret, is refused, with the reason on every rank. The record says `auth=hmac-sha256` or `auth=none`.
- **Without a secret** the job key alone admits a rank (`auth=none`, or `--no-auth` on the launcher): for an isolated development network only. A rank holding a secret refuses a peer without one. `GroupOptions::secret` defaults to none (a launched rank reads `DIST_JOB_SECRET`).
- **The secret is never written down.** `JobSecret`'s `Debug` and `Display` forms show no bytes, and the secret never enters a run record, a log, a checkpoint or saved weights; a test runs a launched job with checkpoints and scans every file it wrote, and the debug forms of the options and the launch configuration, for the secret in hex (both cases) and raw.
- **Open:** only the handshake is authenticated; the collectives' traffic is plain TCP, neither authenticated nor encrypted. Encryption of traffic that leaves a private network is not decided; until then, jobs run inside a private network only.

- **The platform key** (`platform_key()`): the engine's platform key (OS, architecture, the CPU features the kernels select on, the compiler, the gemm kernel's version) plus a fingerprint of the OS maths library: the sha256 of the bits of f32 and f64 `exp`, `ln`, `powf` and `tanh` on fixed inputs that span the ranges the engine reaches. `exp` takes 513 arguments over [−110, 89] (f32) and [−750, 710] (f64): the large negative arguments of a softmax after its max is subtracted, the arguments whose results are subnormal, and those next to and past overflow. `ln` and `powf` take 513 positive values, 64 spread over the subnormals and 449 spread in bit pattern over the whole normal range up to the largest finite value, with `powf` exponents 2, 0.5, −1, 0.37, 3 and −2.5. `tanh` takes 513 arguments over [−20, 20]. Edge arguments are added (±0, the smallest subnormal and normal, the largest finite, −1e4, −1e30, the overflow and subnormal thresholds). A test checks that the inputs reach these ranges. A job whose ranks differ in the key is refused at the rendezvous, before step 0. The per-step weight check (`check_in_sync`) stays as a backstop for anything the fingerprint does not see.
- **The run record** (`ProcessGroup::record()`): the key (with the platform key and its maths fingerprint), the world size, `auth=hmac-sha256` or `auth=none`, and the per-step weight checks: how many passed, `ok` or `failed`, and the last weights' sha256. On this Mac, after a 12-step gate run: `mode=deterministic order=flat-rank backend=tcp platform=macos-aarch64-neon-rustc_1.94.1-matrixmultiply-0.3.11 libm=<sha256> world=2 auth=hmac-sha256 weight_checks=13 ok last_weights=<sha256>`.

## Coordinated checkpoints

- **Agreement.** Every rank hashes its vars, Adam's moments and step count; the hashes must agree (the ranks took identical steps).
- **One writer, verified by every rank.** Every rank builds the checkpoint's bytes itself (`vars.json`, `adam_m.json`, `adam_v.json` in the bit-exact var-map format, and `meta.json` with the step, the world size, both keys, every rank's cursor, the state hash and each file's sha256) and the checkpoint's sha256 (the sha256 of `meta.json`). Rank 0, the single writer, writes `step-<step>/` through a temporary folder and a rename, reads the files back and checks them, and broadcasts the sha256 of what it wrote. Every rank compares it with its own and acknowledges; any mismatch fails the checkpoint on every rank and `latest` is not moved. Then `latest` points at the folder (a rename), and a barrier ends the checkpoint: once any rank is past it, the checkpoint is on disk whole and verified.
- **Copies on other nodes** (`Checkpoint::save_with(…, node_copies = true)`): the first rank of every other node writes the same bytes into its own node-local folder and checks them the same way; a copy is kept only if it holds the same bytes. A folder shared between nodes must not have two writers (leave `node_copies` off there).
- **Resume** (`Checkpoint::load_latest`, called by every rank): every file is checked against its sha256 and the state hash; every rank must have loaded the same checkpoint (the same sha256), so a differing copy stops the resume on every rank; the checkpoint must hold a cursor for every rank, and a rank resumes from its own (`Checkpoint::cursor`); another world size or job key is refused.

## The gate and its results

Run on macOS aarch64 (rustc 1.94.1), CPU reference backend, localhost TCP, at most 4 processes at once. The model: a decoder (vocab 16, width 32, 4 heads, context 8, 2 blocks, MLP 64, dropout 0.1), 2 models on the seed axis, AdamW defaults at lr 0.01, micro-batches of 2 items from a 24-item copy-task set, 12 optimizer steps. Each run is a separate set of processes started by the launcher; the per-step hash of the vars is logged by every rank.

**World sizes against accumulation** (`gate_world_sizes_1_2_4_equal_one_process_with_accumulation`): final weights byte-identical, and every per-step hash identical.

| Global micro-steps | Run | Final weights sha256 |
|---|---|---|
| 2 | 1 process, 2 accumulation micro-steps | `9a24f95f50da08572824c0561890ea3213cc5076274b0cfaffe3bc6abba851d6` |
| 2 | 1 node × 2 processes | same |
| 2 | 2 launchers acting as nodes, on one machine (localhost) | same |
| 2 | distribution off (no rendezvous variables) | same |
| 4 | 1 process, 4 accumulation micro-steps | `02ead67c808b59bde6852388e13d48a89c257b22a8080b95549bc780f2cf1a78` |
| 4 | 2 processes × 2 micro-steps | same |
| 4 | 4 processes × 1 micro-step | same |

**Kill and resume** (`gate_kill_and_resume_equals_the_uninterrupted_run`): a rank is paused at a step, killed by the launcher (SIGKILL), the job stops (the launcher reports the failure and kills the rest), and the same job is relaunched with resume from the last coordinated checkpoint.

| World size | Killed | At step | Checkpoint resumed | Final weights against the uninterrupted run |
|---|---|---|---|---|
| 2 (2 micro-steps) | rank 1 | 7 (checkpoints every 3) | step 6 | identical (`9a24f95f50da08572824c0561890ea3213cc5076274b0cfaffe3bc6abba851d6`), per-step hashes identical |
| 4 (4 micro-steps) | rank 2 | 5 (checkpoints every 2) | step 4 | identical (`02ead67c808b59bde6852388e13d48a89c257b22a8080b95549bc780f2cf1a78`), per-step hashes identical |

Resuming either checkpoint with one process is refused (a resume keeps the world size and job key).

**Sensitivity.** Adding the same 4 micro-batches in another order (`[0, 2, 1, 3]`) changes the weights after 3 steps (`the_gate_can_see_a_changed_order`), and the collectives' test values give other bits in reverse rank order: the identities above are evidence of the order, not of arithmetic that cannot see it.

**Collectives** (`collectives_are_rank_order_exact_at_world_sizes_1_2_4`, ranks as threads over localhost TCP): at W = 1, 2, 4 and with chunks of 2²⁰ and 5 elements, the all-reduce equals the rank-order fold bit for bit on order-sensitive values; broadcast from every root and all-gather return the expected bytes in rank order. `the_gradient_all_reduce_folds_the_global_micro_steps_in_order`: 4 global micro-steps as 1 × 4, 2 × 2 and 4 × 1 (with absent gradients) give the same bits as one process.

**Absent gradients** (`gate_absent_gradients_equal_one_process_with_zeros`): the gate's model plus a var used only in the micro-steps where `(step + j) mod 3 = 1` (j the global micro-step), whose gradient holds −0 and +0; at 2 micro-steps every third step leaves it untouched on every rank. 9 steps. The reference is one process that puts zeros in place of the absent gradients and adds them with the engine's own tensor additions (no `GradSum`).

| Global micro-steps | World sizes | Final weights (all equal to the reference) |
|---|---|---|
| 2 | 1, 2 | `7c61a7994ceb0d787ef5ca45a5fd604075a5d3f21c1cf84add0cd59ee590d67c` |
| 4 | 1, 2, 4 | `0ad5e705faca0758a2d0fd8dc06a82106a99c5c3f4d1962417e0e2f27f79d3e7` |

`absent_gradients_count_as_zeros` checks the same at the gradient level, bit for bit, for absent-first, absent-last and absent-everywhere cases.

**An idle weight** (`gate_an_idle_weight_is_left_alone_like_one_process`): the gate's model plus a weight no micro-step on any rank ever uses, trained by AdamW with decoupled decay 0.1 in both orders, 4 global micro-steps, 6 steps. One process (tensor additions, no `GradSum`) leaves the weight's bits untouched, decay included, and every world size matches it byte for byte.

| Decay order | World sizes | Final weights (all equal to one process) |
|---|---|---|
| after the step | 1, 2, 4 | `2a30f578eadf26ff812e297ddea86d0228457322d3e839197e015ce0be08638e` |
| before the step (torch) | 1, 2, 4 | `4decceea205f0f8a6581050428afc6932b0aca0390ee476137863d3df938db53` |

**Checkpoint copies** (`checkpoint_copies_on_other_nodes_hold_the_same_bytes_and_resume_checks_them`, 2 nodes × 1 process as threads, a folder per node): the copy on node 1 is byte-identical to rank 0's; both ranks resume from their own copy to the same vars; one changed byte in node 1's copy stops the resume on both ranks.

**Refusals and failures:** another job key, platform key, world size or job secret (or none against one) at the rendezvous (refused on every rank, with the reason), mismatched collectives, a rank that leaves (its peers fail at once), a later rank that pre-added its micro-steps, a failing rank under the launcher (the job stops; the others are killed), a changed byte in a checkpoint (sha256 mismatch).

**The example** (`examples/ddp_copy_task.rs`, 200 steps, 4 global micro-steps of 8 items): final weights sha256 `7d4c24b808b50f964754c8c2e6a674ba7b82e91af96023a9649ad0ea22be6012` alone, under the launcher with 2 processes and with 4, and with 2 launchers acting as nodes of 2 processes each on one machine (a shared secret file), on every rank.

The whole gate (27 distributed tests, plus the HMAC vectors) runs in about 5 s: `cargo test --release --lib nn::dist`.

## GPU ranks (unreleased)

Status: built; tested on one machine per GPU kind (the configurations and results below); multi-GPU, multi-machine and datacenter runs untested (no multi-GPU machine available yet). GPU results are keyed apart from CPU results and are never compared byte for byte with them.

**What changes.**
- **One device per rank.** `DistEnv::device(kind, ranks_per_device)` gives the rank's device from its rank on the node (`LOCAL_RANK`), as torchrun's `cuda:LOCAL_RANK`: `Cuda(LOCAL_RANK / ranks_per_device)`, `Metal(…)` likewise, or `Cpu(Reference)`. With `ranks_per_device` 1 each rank has its own GPU; with more, that many ranks share one (how the gates below run on one GPU). `GroupOptions::device` is the rank's device; `data_parallel_step` refuses vars that are not on it.
- **The CUDA backend runs on `Cuda(i)`**, one device per process: the first use fixes the process's device (`Cuda(0)` unless another is named first); another device afterwards is refused. A device the machine does not have is refused without fixing anything. Only `Cuda(0)` was available to test.
- **The sum is 0.3.0's.** Each micro-step's gradients are copied from the device to the host as f32, summed by `GradSum` in the flat rank order (rules 1–3, unchanged), and the sum goes back to the device, where every rank takes the same optimizer step. Correct, not fast: overlap and bucketing are later work and must keep the order.
- **The key.** `device_key(device)`: none for `Cpu(Reference)` (the CPU key is 0.3.0's, unchanged), else `device=` and `tensor::gpu_platform_key(device)`, which is `gpu_key(device)` (unchanged) followed by what the results depend on outside this crate:
  - CUDA: the GPU's name and compute capability, the CUDA version the driver supports (`driver…`), NVRTC's version, the kernel options, the kernel source's hash, and the driver's release (`driver-release-…`, read through NVML, which ships with the driver and is now linked with the `cuda` feature; the driver compiles the kernels' PTX). On the test machine: `device=cuda-NVIDIA_GeForce_RTX_4060_Laptop_GPU-sm89-driver13040-nvrtc12.6-nofastmath-fmadoff-kernels-<16 hex digits of the kernel source's sha256>-driver-release-617.14`.
  - Metal: the GPU's name, the kernel options and source hash, the highest Apple and Metal GPU families it supports, its architecture's name, and the OS version and build (the OS compiles the kernels' source). On the test machine: `device=metal-<the GPU's name>-mathmode-safe-kernels-<16 hex digits of the kernel source's sha256>-family-apple7-metal4-mac2-arch-applegpu_g13s-macos-Version_27.0.1__Build_26A434_`.
  - The rendezvous compares the whole platform key, so a job whose ranks run on different kinds or models of device, or under different drivers, is refused before step 0.
- **Resume** also refuses a checkpoint taken under another group key (another platform or device: a CPU checkpoint on a GPU, another GPU model or driver), as it refuses another world size or job key.
- **The fast CPU backend** is not a device of the deterministic mode: `device_key(Cpu(Fast))` is an error.

**The prerequisite: one GPU, run to run** (`gpu_training_is_byte_identical_run_to_run`). The same training, run twice in two processes, is byte-identical on the GPU: every per-step hash of the vars, the per-seed losses and the final vars of three configurations (the gate's decoder with dropout 0.1 through `data_parallel_step` with 2 micro-steps, 12 steps; a mixture-of-experts decoder with top-2 routing and capacity, whose gather, scatter and index adds and sort run on the device, 12 steps; a wider decoder, d 128, MLP 512, context 64, whose products take the 128-tile and, on CUDA, the split-k matmul paths, 6 steps). The kernels used in training were read for nondeterminism: no atomic operations (a test checks the CUDA and Metal sources for them), every sum in a fixed order (split-k folds its blocks in order), kernel and tile choices by shape only (no timing, no autotuning), cuBLAS not used by tensors, fast math off. No kernel source changes.

| Device | steps.txt (per-step hashes and losses) | Gate decoder | MoE decoder | Wide decoder |
|---|---|---|---|---|
| CUDA (Windows x86_64, RTX 4060 Laptop GPU, driver 617.14 (CUDA 13.4 driver API), NVRTC 12.6) | `3b5921ae3902abc7365a77d220e5f26403ef0e733c3235a9f94e9eaf5da2dbd7` | `3263b99095eded48b1d0cc9c69d2492b56d9dfe38573ed35f797a6132339ff49` | `e576bd23c6877557157cd640e50d34e8fae238f8bba7535d10466f1d2602f38b` | `ff7107fe1904fbcb7b58ec25e8e8c38158f3e36158db7f4e43e3ecaf18805230` |
| Metal (macOS 27.0.1 aarch64, an Apple GPU of the Apple7 and Metal 4 families, architecture applegpu_g13s) | `317b64881d59fd7e8370b9643f9212cf5dae23ab573a020f0359b63439420e6e` | `e261aa4cd3b1b6b9eeb4e1ba4f8699eb1f648a1c2032d9028d08a03b9918306f` | `ce9bd640bdcda8c5b47a8b4c2348544573d4af12ab2c496f72154b8b07e1b76b` | `361eb7d669c4d492ca7f3201d14f0cb6f3f7d7523ec8ba6b318a38c59ca12287` |

Each value is the same in both runs.

**The GPU gates** (the gate's model and data as above, AdamW, 12 steps, per-step weight checks on every rank; the ranks are processes started by the launcher, all on the machine's one GPU):

`gpu_gate_two_ranks_on_one_gpu_equal_one_gpu_with_accumulation`: final weights byte-identical, and every per-step hash identical.

| Global micro-steps | Run | CUDA final weights sha256 | Metal final weights sha256 |
|---|---|---|---|
| 2 | 1 process on the GPU, 2 accumulation micro-steps | `3263b99095eded48b1d0cc9c69d2492b56d9dfe38573ed35f797a6132339ff49` | `e261aa4cd3b1b6b9eeb4e1ba4f8699eb1f648a1c2032d9028d08a03b9918306f` |
| 2 | 2 processes sharing the GPU, 1 micro-step each | same | same |
| 2 | distribution off (no rendezvous variables) | same | same |
| 4 | 1 process on the GPU, 4 accumulation micro-steps | `4021208e74c5970b4ba0399cda524de9c497856635ab4a673066ebee72965d04` | `4d1adffd836bbd93cd25e3522f02c3f856feaf6c02b64ecd9afc0086767fdc21` |
| 4 | 2 processes sharing the GPU, 2 micro-steps each | same | same |

`gpu_gate_kill_and_resume_equals_the_uninterrupted_run`: 2 processes sharing the GPU, 2 micro-steps, checkpoints every 3 steps; rank 1 paused at step 7 and killed by the launcher; the job stops; the relaunched job resumes from step 6. Final weights and every per-step hash equal the uninterrupted run's (CUDA `3263b99095eded48b1d0cc9c69d2492b56d9dfe38573ed35f797a6132339ff49`, Metal `e261aa4cd3b1b6b9eeb4e1ba4f8699eb1f648a1c2032d9028d08a03b9918306f`). Resuming that checkpoint with one process (another world size) is refused, and so is resuming it with 2 processes on the CPU (another key).

`gpu_a_job_mixing_gpu_and_cpu_ranks_is_refused`: rank 0 on the GPU and rank 1 on the CPU: the rendezvous refuses the job (a mixed job) before step 0.

These gates show the order of the sum and the checkpoint path on a GPU. They do not show anything about several GPUs: both ranks run on one GPU, and `Cuda(i)` for i > 0 was not available.

## Single-device behaviour is unchanged

- No existing file changes behaviour. The changes outside `nn::dist`: the module line in `nn/mod.rs`, a new error variant `NnError::Dist` (kind "dist"), two examples, this note and a README bullet.
- All 88 existing library tests and the README doctests pass.
- Six training configurations (decoder, MoE decoder, decoder with dropout; each with AdamW and SGD with momentum; 60 steps, 3 models on the seed axis) give byte-identical final vars and logits built against 0.2.0 and against this branch; the copy-task example's output is identical.
- `data_parallel_step` with the single-process group and one micro-step equals `nn::train_step` bit for bit (vars and Adam moments, `one_process_with_one_micro_step_equals_the_plain_trainer`).
- **GPU ranks (unreleased):** the same six configurations give byte-identical final vars and logits built against 0.3.0 and against the GPU-ranks change, on the CPU reference backend (macOS aarch64 and Windows x86_64), on CUDA (`Cuda(0)`, Windows) and on Metal (macOS). The CPU gates above give the same hashes as 0.3.0 (on Windows every one of them; on the Mac those that run in at most 2 processes at once, see the machine note in the change log), and the CPU key is unchanged. The other changes outside `nn::dist`: `tensor::gpu_platform_key`, the CUDA backend on `Cuda(i)` (one device per process) and NVML linked with the `cuda` feature; no kernel source changes.

## For the release notes

- adds the variant NnError::Dist
- public API additions (`nn::dist`, among them `GroupOptions` with public fields `platform` and `secret`, `JobSecret`, `platform_key`, `LaunchConfig::secret`): a public API change, so the release carrying them needs a minor version bump
- unreleased (GPU ranks): `GroupOptions` gains the public field `device` (a breaking change for struct literals without `..Default::default()`), `DeviceKind`, `DistEnv::device`, `device_key`, `ProcessGroup::device` and `ProcessGroup::platform`, `tensor::gpu_platform_key`; the `cuda` feature links NVML; `Cuda(i)` with i > 0 is accepted (one device per process) where 0.3.0 refused it; a resume under another group key is refused

## Known limits and next steps

- **f32 gradients only.** f64, f16 and bf16 gradients are refused by `GradSum`. bf16 with f32 master weights comes with mixed precision.
- **Tested on one machine per device kind.** GPU gradients are copied to the host for the sum (correct, not fast). The GPU gates ran two processes on one GPU; multi-GPU and multi-machine runs need the later gates (1 node × 2 GPUs, 2 nodes × 1 GPU, against 1 GPU × 2 micro-steps).
- **The OS maths library.** softmax, log_softmax and `powf` call the platform's `exp`, `log` and `pow`, whose last bits may differ between operating systems or library versions. The gate runs on one machine, so it is unaffected. Between machines, the rendezvous refuses ranks whose maths library fingerprint differs (the fingerprint samples the functions on fixed inputs; it can miss a difference elsewhere, which `check_in_sync` then reports as an error rather than letting the ranks drift). Mixed-platform jobs need the engine's own implementations of these functions.
- **Throughput.** The running sum passes the ranks one after another (latency rises with W) and the last rank sends the total to each rank (its traffic rises with W). Both keep the order; a tree or ring broadcast of the total, bucketing and overlap with the backward pass are later work and must not change the order of the additions.
- **Memory.** A rank after the first keeps its k micro-step gradients until the running sum arrives (k copies of the gradients; with one micro-step per rank, none extra).
- **Checkpoints.** Rank 0 is the single writer; node copies are optional and checked. The optimizer state saved is Adam's; SGD's momentum buffers are not yet in the checkpoint.
- **Not yet built:** reduce-scatter, sharded optimizer state and parameters (FSDP-style), elastic restart, a fast mode, generic callbacks (batch seen, checkpoint reached), NCCL.
