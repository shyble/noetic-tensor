# Distributed training: deterministic data parallelism, and a fast mode

Released in 0.3.0 (9 October 2026) for the CPU. Released in 0.4.0 (10 October 2026): ranks on a GPU, one device per rank (see "GPU ranks" below); built; tested on one machine per GPU kind (CUDA: up to four processes sharing a single CUDA GPU on Windows; Metal: two processes sharing one Apple GPU on macOS; shown: byte-identical weights to one GPU with accumulation, with absent gradients, an idle weight under decay, clipping, and a killed and resumed run under AdamW and under SGD with momentum); multi-GPU, multi-machine and datacenter runs untested. Released in 0.5.0 (15 October 2026): sharded data parallelism and the gate as an example binary with one device per rank (see those sections below), and the engine and wire-protocol versions at the rendezvous; built; tested on one machine per device kind; multi-GPU, multi-machine and datacenter runs untested. Released in 0.6.0 (15 October 2026): a fast mode, opt-in, under its own key and never pooled with the deterministic mode (see "Fast mode" below), and mixed precision (see `doc/mixed_precision.md`); built; tested on one machine per device kind; multi-GPU, multi-machine and datacenter runs untested. After 0.6.0's release the deterministic mode was tested on two GPUs in one machine and on two machines with one GPU each (see "Two GPUs in one machine, and two machines"); the fast mode and mixed precision were not.

## What it adds

`noetic::nn::dist`: data-parallel training across processes (tested on one machine, and on two machines with one GPU each; larger configurations untested), with an API in the shape of torch.distributed, DDP and torchrun. Standard library only (`std::net`, `std::process`); no new dependency.

| Item | What it does |
|---|---|
| `DistEnv` | The process's place in the job, from torchrun's variables: `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE`, `GROUP_RANK`. Without `RANK` and `WORLD_SIZE`, distribution is off. |
| `ProcessGroup` | The rendezvous, a full mesh of TCP connections, and the collectives: `barrier`, `broadcast`, `all_gather`, `all_reduce_f32`, `all_reduce_grads`. `ProcessGroup::single()` (and `from_env` with distribution off) opens no socket; its collectives are the identity. |
| `GradSum`, `Reduce` | The one ordered gradient sum, used by local accumulation and by the all-reduce. |
| `data_parallel_step` | One optimizer step: this rank's micro-steps, the all-reduce, the same optimizer step on every rank. With the single-process group it is gradient accumulation. |
| `data_parallel_grads` | Since 0.4.0. The summed gradients of a step, the same on every rank, for a caller that transforms them (clips them) before its optimizer steps. |
| `check_in_sync`, `state_hash` | Every rank holds the same vars, bit for bit (all-gather of a sha256 of the saved vars). |
| `Shard`, `ShardSpec` | The distributed sampler: each rank's micro-batches, from (seed, rank, world size) only; a sha256 per shard. |
| `Checkpoint`, `CheckpointOptimizer` | Coordinated checkpoints: vars, the optimizer's whole state (Adam's moments and step count; since 0.4.0, SGD's momentum buffers), every rank's loader cursor. |
| `LaunchConfig`, `spawn`, `run`, `Job` | A torchrun-style launcher: N processes per node, rendezvous variables and the job's secret set, a log per rank; the first failure stops the job. |
| `JobSecret`, `platform_key` | The per-job secret of the authenticated handshake; the platform key every rank must share. |
| `GroupOptions::new`, `with_*` | Since 0.4.0. `GroupOptions` is `#[non_exhaustive]`: made with `new` (or `default`) and the `with_*` methods, so later fields do not break callers; its fields stay public to read and set. |
| `DeviceKind`, `DistEnv::device`, `GroupOptions::device`, `device_key` | Since 0.4.0. The rank's device from its rank on the node (torchrun's `cuda:LOCAL_RANK`), the group's device, and the device's part of the platform key (`device=…`, none for the CPU reference backend). |
| `examples/launch.rs` | The launcher as a command: `launch --nproc-per-node N [--nnodes M --node-rank I --master-addr A --master-port P] [--log-dir D] [--secret-file F | --no-auth] -- <command> [args]`. |
| `examples/ddp_copy_task.rs` | The copy task trained data-parallel; the same weights at 1, 2 and 4 processes. |
| `ShardedDataParallel`, `ShardOptions`, `ShardGrads`, `ShardLayout` | Since 0.5.0. Sharded data parallelism in the shape of ZeRO's stages and FSDP, with the weights of data parallelism bit for bit; consolidated and sharded checkpoints (see "Sharded data parallelism"). |
| `examples/ddp_gate.rs` | Since 0.5.0. The gates' training as an example binary, one device per rank from `LOCAL_RANK` (see "The gate as an example binary"). |
| `mixed_data_parallel_step`, `mixed_data_parallel_grads`, `mixed_sharded_step`, `MixedStep`, `mixed_record`, `ProcessGroup::matmul_precision` | Since 0.6.0. Mixed precision: the products at `MatmulPrecision::Half`, f32 vars and optimizer state, a static loss scale (`nn::LossScale`); in either mode, data-parallel or sharded; its own key (see `doc/mixed_precision.md`). |
| `DistMode`, `FastOptions`, `GroupOptions::with_mode`, `ProcessGroup::mode`, `is_fast`, `require_deterministic` | Since 0.6.0. The fast mode: per-rank sums, gradient buckets, a ring all-reduce (a ring reduce-scatter for sharding) and communication overlapped with the backward pass; its own key; not reproducible bit for bit (see "Fast mode"). |

A training loop:

```rust
use noetic::nn::dist::{check_in_sync, data_parallel_step, GroupOptions, ProcessGroup, Reduce, Shard, ShardSpec};

let mut group = ProcessGroup::from_env(GroupOptions::new("my run"))?;
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

## Determinism rules (deterministic mode, the default)

1. **One order for the sum.** A step's gradient is the left fold over its global micro-steps, `((g₀ + g₁) + g₂) + …`, elementwise in f32. Rank r's i-th of k micro-steps is global micro-step `r·k + i`. The fold starts from the first micro-step's gradient (never from an extra zero, which would turn −0 into +0). A var without a gradient in a micro-step (it took no part: a branch not taken, a var unused by that batch) counts as zeros, so the sum is bit for bit the one a single process gets by putting zeros in place of the absent gradients and adding them (`x + 0` included, which turns −0 into +0). A var no micro-step touched stays `None`, and the optimizer skips it, as it does today and as torch does for a parameter unused on every rank; AdamW's decoupled decay is skipped with it, exactly as in one process.
2. **One routine.** `GradSum` is the only place gradients are added, for local accumulation and for the all-reduce. `Reduce::Mean` divides by the number of global micro-steps as `Tensor::div_scalar` does, and not at all when there is one (so a single micro-step passes through bit for bit).
3. **Rank order across the network.** The all-reduce passes the running sum along the ranks: rank 0 sends its sum to rank 1, which folds its own micro-steps in, in order, and sends it on; the last rank sends the total to every other rank once it has all of it. Large sums travel in chunks so the ranks work at the same time; chunking changes the pipelining, never the order of the additions. A rank after the first therefore keeps its micro-steps apart until the running sum reaches it (`ProcessGroup::grad_sum` gives that kind of sum; pre-adding them is refused).
4. **Identical optimizer steps.** Every rank applies the same optimizer to the same summed gradient, so the vars stay identical without being sent. `check_in_sync` verifies it, and every coordinated checkpoint verifies it again.
5. **No timing in the result.** No compression, no quantised or asynchronous reductions, no arrival-order sums. Every message carries its collective's operation and sequence number; ranks that call different collectives fail at once.
6. **The key and pooling.** `ProcessGroup::key()` is the reproducibility key, `mode=deterministic order=flat-rank backend=tcp platform=<platform key>`; on a GPU the platform key ends with the device's part, `device=<GPU platform key>` (see "GPU ranks"), so CPU and GPU results, and two GPU models or drivers, never share a key. The world size is not part of it: by rules 1–3 the weights depend on the number of global micro-steps, not on how they are split among ranks (the gate below shows it), so runs at different world sizes may be pooled on the same platform key. The world size stays in every run's record (`ProcessGroup::record()`, and the checkpoint's meta). Before pooling on a new (platform, device) pair, a short run with per-step weight-hash checks at world sizes 1, 2 and 4 must show the identity there; a GPU needs its own showing. A resume still keeps the checkpoint's world size and job key (it refuses others): pooling compares finished runs, it does not move a run between world sizes. The refusal is lifted only after a gate shows a resume at another world size byte-identical to the uninterrupted run.
7. **Order variants.** The flat order is the default and the only one built. A per-rank-first order (each rank pre-adds its micro-steps, then the ranks' sums are added in rank order) would save the deferred micro-steps' memory but makes the bits depend on the world size; if it is ever added, it is an opt-in whose key includes the world size, and it is never pooled across world sizes.
8. **Randomness inside a micro-step** (dropout) must be keyed by the global micro-step, not by the rank (the gate's model uses dropout 0.1 this way).
9. **Data.** Item p of the global stream is `perm_e[p mod n]` with e = p div n, where `perm_e` is a Fisher–Yates permutation from `rng::derive(seed, "shard epoch e")` (u64 draws, platform-independent). Global micro-batch j of step s starts at item `(s·M + j)·b`. Rank r takes micro-batches `r·k .. (r+1)·k`, so the ranks' micro-batches in rank order are the single process's. The loader cursor is the step number. `Shard::hash` is a sha256 of the shard's key (seed, items, micro-batch, micro-steps, rank, world size) and every item it takes; the training ranks all-gather their shard hashes and check every rank's against their own computation.
10. **The fast mode** (since 0.6.0; see "Fast mode" below) is opt-in (`GroupOptions::mode`), carries its own key (`mode=fast …`), and is never pooled with or compared byte for byte against this mode; a job mixing the modes is refused at the rendezvous. Rules 1–9 are this mode's alone.

## Rendezvous and failure

- Rank 0 listens on `MASTER_ADDR:MASTER_PORT`. Every other rank connects, answers rank 0's challenge (below), and sends its rank, world size, job key, platform key and the address of its data listener (on the interface it reached rank 0 through). Rank 0 checks that the ranks are distinct and agree on the world size, the job key and the platform key, then sends the address table, or the reason for refusing the job, to every rank. Ranks connect to every lower rank, accept every higher one, and a barrier ends the rendezvous.
- A lost peer is an error on the next message (end of stream), not a hang; a silent peer fails after `GroupOptions::timeout` (600 s by default).
- The launcher stops the whole job when any rank fails or is killed, and names the rank.
- **The authenticated handshake.** The launcher gives every rank the job's secret (`JobSecret`, at least 32 bytes, in `DIST_JOB_SECRET`; a single-node launch makes a fresh one, a multi-node job passes the same `--secret-file` on every node). At the rendezvous and on every link of the mesh each side sends a fresh 32-byte random challenge and the other answers with HMAC-SHA256(secret, label · challenge · its message), computed in-house on the existing sha256 (checked against RFC 4231's vectors). A rank that cannot answer, or answers for another secret, is refused, with the reason on every rank. The record says `auth=hmac-sha256` or `auth=none`.
- **Versions** (since 0.5.0; wire protocol 3 since 0.6.0). Every rank's hello says its engine version and wire-protocol version after its platform key (`wire_version()`: `engine=0.6.0 wire=3`; 0.5.0 said `engine=0.5.0 wire=2`). Rank 0 refuses a rank that says another engine or protocol, or none (an engine of 0.4.0 or earlier), before step 0, with the reason on every rank (`the_rendezvous_refuses_another_engine_or_wire_version`); a 0.4.0 rank 0 refuses a 0.5.0 rank as a mixed platform. The versions are not part of the group's key.
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

## GPU ranks (0.4.0)

Status: built; tested on one machine per GPU kind (the configurations and results below), and after 0.6.0's release on two L4 GPUs in one machine and on two machines with one L4 each (see "Two GPUs in one machine, and two machines"); larger configurations untested. GPU results are keyed apart from CPU results and are never compared byte for byte with them.

**What changes.**
- **One device per rank.** `DistEnv::device(kind, ranks_per_device)` gives the rank's device from its rank on the node (`LOCAL_RANK`), as torchrun's `cuda:LOCAL_RANK`: `Cuda(LOCAL_RANK / ranks_per_device)`, `Metal(…)` likewise, or `Cpu(Reference)`. With `ranks_per_device` 1 each rank has its own GPU; with more, that many ranks share one (how the gates below run on one GPU). `GroupOptions::device` is the rank's device; `data_parallel_step` refuses vars that are not on it.
- **The CUDA backend runs on `Cuda(i)`**, one device per process: the first use fixes the process's device (`Cuda(0)` unless another is named first); another device afterwards is refused. A device the machine does not have is refused without fixing anything. `Cuda(1)` was tested on two L4 GPUs in one machine after 0.6.0's release: the gate's training on `Cuda(1)` equalled `Cuda(0)`'s, and two ranks, one per GPU, equalled one GPU with accumulation (see "Two GPUs in one machine, and two machines"). `Cuda(i)` for i > 1 is untested.
- **The sum is 0.3.0's.** Each micro-step's gradients are copied from the device to the host as f32, summed by `GradSum` in the flat rank order (rules 1–3, unchanged), and the sum goes back to the device, where every rank takes the same optimizer step. Correct, not fast: overlap and bucketing are later work and must keep the order.
- **The key.** `device_key(device)`: none for `Cpu(Reference)` (the CPU key is 0.3.0's, unchanged), else `device=` and `tensor::gpu_platform_key(device)`, which is `gpu_key(device)` (unchanged) followed by what the results depend on outside this crate:
  - CUDA: the GPU's name and compute capability, the CUDA version the driver supports (`driver…`), NVRTC's version, the kernel options, the kernel source's hash, and the driver's release (`driver-release-…`; the driver compiles the kernels' PTX). The release is read through NVML, the driver's management library, loaded at run time (`nvml.dll`, `libnvidia-ml.so.1`), not linked: the `cuda` feature builds, and a process trains on one GPU, where NVML is missing (some containers lack it). There `device_key` is an error: a GPU key is never made without the driver's release (`cuda_a_missing_nvml_refuses_the_key_but_not_the_device`). On the test machine: `device=cuda-NVIDIA_GeForce_RTX_4060_Laptop_GPU-sm89-driver13040-nvrtc12.6-nofastmath-fmadoff-kernels-<16 hex digits of the kernel source's sha256>-driver-release-617.14`.
  - Metal: the GPU's name, the kernel options and source hash, the highest Apple and Metal GPU families it supports, its architecture's name, and the OS version and build (the OS compiles the kernels' source). On the test machine: `device=metal-<the GPU's name>-mathmode-safe-kernels-<16 hex digits of the kernel source's sha256>-family-apple7-metal4-mac2-arch-applegpu_g13s-macos-Version_27.0.1__Build_26A434_`.
  - The rendezvous compares the whole platform key, so a job whose ranks run on different kinds or models of device, or under different drivers, is refused before step 0.
- **Resume** also refuses a checkpoint taken under another group key (another platform or device: a CPU checkpoint on a GPU, another GPU model or driver), as it refuses another world size or job key.
- **Recording the kernels' build.** `tensor::cuda::kernel_ptx_and_cubin(cc)` returns NVRTC's PTX for the kernel source (byte for byte what the backend loads) and the CUBIN NVRTC makes for the real architecture, for a run's records. On the RTX 4060 Laptop GPU (sm89, NVRTC 12.6): PTX 324062 bytes, CUBIN 243488 bytes (`nvrtc_gives_the_backends_ptx_and_a_cubin`). The CUBIN is NVRTC's machine code, which need not be the driver's compilation of the PTX the backend runs.
- **The fast CPU backend** is not a device of the deterministic mode: `device_key(Cpu(Fast))` is an error. It is a device of the fast mode only (`device=cpu-fast` in that key; see "Fast mode").
- **The whole optimizer state is checkpointed, or the resume is refused.** `Checkpoint::save` takes any `CheckpointOptimizer`: Adam (both moments and the step count, written byte for byte as in 0.3.0) or SGD (its momentum buffers, `sgd_momentum.json`, and `"optimizer": "sgd"` in the meta file). `Checkpoint::restore` refuses an optimizer of another kind than the one whose state the checkpoint holds, so a run never resumes without its optimizer state.
- **Clipping.** `data_parallel_grads` returns the step's summed gradients, the same on every rank; a caller clips them (`clip_grad_norm_per_seed`) and steps, as every rank then does identically. `data_parallel_step` is `data_parallel_grads` followed by the optimizer's step.

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
| 4 | 4 processes sharing the GPU, 1 micro-step each (VRAM allowed it on the 8 GB GPU) | same | not run (the Mac's 2-process cap) |

`gpu_gate_kill_and_resume_equals_the_uninterrupted_run`: 2 processes sharing the GPU, 2 micro-steps, checkpoints every 3 steps; rank 1 paused at step 7 and killed by the launcher; the job stops; the relaunched job resumes from step 6. Final weights and every per-step hash equal the uninterrupted run's (CUDA `3263b99095eded48b1d0cc9c69d2492b56d9dfe38573ed35f797a6132339ff49`, Metal `e261aa4cd3b1b6b9eeb4e1ba4f8699eb1f648a1c2032d9028d08a03b9918306f`). Resuming that checkpoint with one process (another world size) is refused, and so is resuming it with 2 processes on the CPU (another key).

`gpu_a_job_mixing_gpu_and_cpu_ranks_is_refused`: rank 0 on the GPU and rank 1 on the CPU: the rendezvous refuses the job (a mixed job) before step 0.

`gpu_gate_absent_gradients_and_an_idle_weight_equal_one_gpu`: 0.3.0's absent-gradient and idle-weight gates through the GPU path. The reference is one process on the same GPU that puts zeros in place of the absent gradients and adds them with the engine's own tensor additions and `div_scalar` on the GPU (no `GradSum`); the absent var's gradient holds −0 and +0, and at 2 micro-steps every third step has none at all; the idle weight is trained by AdamW with decoupled decay 0.1 in both orders and keeps its initial bits, decay included.

| Gate | World sizes | CUDA final weights (all equal to one GPU) | Metal final weights (all equal to one GPU) |
|---|---|---|---|
| absent, 2 micro-steps, 9 steps | 1, 2 | `ed14a6550cd5773878d04a6516ab7f3c5a9ab9df6768db24c6f7f8d0c907aa1c` | `055998bd589d27e9f2755f18d42c47564edb6b530010d30ed5735011ae4ddc76` |
| absent, 4 micro-steps, 9 steps | 1, 2, 4 (Metal: 1, 2) | `5bae8705a1f990ecfdbae8f8f19619b914347bdd753de2d0fb092cfc29ff6f60` | `0ef5c715415531e36d293ef4e38a109ae1026c3e3ec99a631b5f2a162e0fcf41` |
| idle weight, decay after the step, 4 micro-steps, 6 steps | 1, 2, 4 (Metal: 1, 2) | `54f777084d76890aa834b87be8c76a77a5861386c125d901935e3e0fdcf7bc2f` | `91867037aef3b6445fb2d246a188a2b76ead652ef5b4b8586911a7f76b3ed38a` |
| idle weight, decay before the step (torch) | 1, 2, 4 (Metal: 1, 2) | `136006461c91ed08f4845a58562cc4c5a433fb40507c805f593c6d8f76d33da0` | `f51d05efa6242d0828b2710d76ab5ab39b6e341345b70e6df770b703356c7367` |

`gpu_gate_clipping_world_sizes_equal_one_gpu`: every rank clips the summed gradients per seed to norm 1.0 before AdamW steps (the clip engaged on every one of the 12 steps); final weights and every per-step hash equal one GPU's.

| Global micro-steps | World sizes | CUDA final weights | Metal final weights |
|---|---|---|---|
| 2 | 1, 2 | `d481ddb36d6a9c3f1a169e67d6eda1d1b6fa6a93edd3944a4d1edf30f30de873` | `6f11db4d82361476e986921e29c8e9ab3708f41c6b6f4cef51536a41ff84ccdb` |
| 4 | 1, 2, 4 (Metal: 1, 2) | `561ebc59ea9cce995956425c60e59b15d3a1049428987ffb115ab00589ff93e3` | `b43b6924c7fbe8d7ad8532ed0263a0771f81f68d3cdb6daeb72a5a3fdef84648` |

The same gate on the CPU (`gate_clipping_world_sizes_equal_one_process`): macOS aarch64 `382d20b6b7c970187cabb9909918c78cc7d4a23695de51d9d1c846fddacb99c0` (2 micro-steps) and `5d4fa2a91c75d1803519b3674cd7acf231568b7e461ca71064fa0d1180f936b2` (4, world sizes 1 and 2); Windows x86_64 `aa1b37445adbfc1e269380a7a84ae4cb0b8a310558ca89c0dc723f714fe215c3` (2) and `dae9d10a3171b13c08f938804a539f946fc93f88bd54895c403bcf2b7c452034` (4, world sizes 1, 2 and 4).

`gpu_gate_kill_and_resume_under_sgd_with_momentum`: as the resume gate above, under SGD (lr 0.05, momentum 0.9, weight decay 1e-3): the resumed run's final weights and every per-step hash equal the uninterrupted run's (CUDA `7650ce860e8db697d96e771ece5d20fca65c534d33353d6cc31fcb1ebf18d39c`, Metal `fddba9d2e0cdc2d4b5a94144349bd7f41aea395b86cdeede8de490aaa04a877e`; on the CPU, `gate_kill_and_resume_under_sgd_with_momentum`, macOS aarch64: `9f5baec7e30edc422ce97941c1f600da9ed643cb70892b1821b1a74660417694`). An AdamW run resuming the SGD checkpoint is refused. `an_sgd_checkpoint_resumes_exactly_and_another_optimizer_is_refused` shows that a resume without the momentum buffers would give other weights, so the gate can see a lost buffer.

**Where every gate ran.** Windows x86_64 (CUDA build) ran every gate with no skips, world size 4 included (the 4-process runs in a separate short session, as the machine was shared). The Mac was shared with a long benchmark, and its jobs were capped at 2 threads: there the runs needing 4 processes at once were skipped (`DIST_TEST_MAX_PROCS=2`, a development switch; each skip is printed), so the Metal results above are at world sizes 1 and 2.

These gates show the order of the sum and the checkpoint path on a GPU. They do not show anything about several GPUs: all ranks run on one GPU, and `Cuda(i)` for i > 0 was not available.

## The gate as an example binary: one device per rank (0.5.0)

Released in 0.5.0 (15 October 2026). Status: built; tested on one machine per device kind (the CPU reference backend on macOS aarch64 and Windows x86_64, a single CUDA GPU on Windows shared by up to four processes, one Apple GPU on macOS shared by two; shown: the reference's weights and every per-step hash at every configuration, a resume from a checkpoint); multi-GPU, multi-machine and datacenter runs untested.

The GPU gates above run every rank of a node on its first GPU, and `examples/ddp_copy_task.rs` runs on the CPU only. `examples/ddp_gate.rs` is the gates' training as an example binary that the launcher (`examples/launch.rs`) starts on one or several machines, each rank on the device its rank on the node gives (`DistEnv::device`: `--ranks-per-device 1` is torchrun's `cuda:LOCAL_RANK`, one GPU per rank; K ranks per device share a GPU). Run without rendezvous variables it is the reference: one process on one device accumulating the M global micro-steps.

- The gate's model, data, dropout keying and optimizers, exactly as the gates: `--micro M`, `--steps N` (12), `--opt adamw|sgd`, `--clip X`, `--absent`, `--idle after|torch`.
- Every rank prints the vars' sha256 after each step (checked equal on every rank by `check_in_sync`); rank 0 prints the final weights' sha256 and the run's record (key, world size, `auth`, weight checks). `--out DIR` writes `rank-<r>.steps`, `final.json` and `record.txt`.
- Coordinated checkpoints (`--ckpt-every E`, `--ckpt-dir D`, `--resume`, `--node-copies`), `--pause R:S` (a rank waits before step S, to be killed), `--cpu-ranks R,…` (a mixed job, for the refusal), `--job-key K`.
- Every rank checks every rank's shard hash for the run's steps, and rank 0 prints its own.

```sh
E=target/release/examples                     # cargo build --release --features cuda --examples
$E/ddp_gate --device cuda --micro 2           # the reference: 1 GPU × 2 micro-steps
$E/launch --nproc-per-node 2 -- $E/ddp_gate --device cuda --micro 2                        # one GPU per rank
$E/launch --nproc-per-node 2 -- $E/ddp_gate --device cuda --micro 2 --ranks-per-device 2   # both on GPU 0
```

On the CPU it reproduces the gates' recorded weights (`9a24f95f50da08572824c0561890ea3213cc5076274b0cfaffe3bc6abba851d6` at 2 micro-steps, `02ead67c808b59bde6852388e13d48a89c257b22a8080b95549bc780f2cf1a78` at 4, the clipping and SGD gates' hashes), and on each GPU the GPU gates' (CUDA `3263b99095eded48b1d0cc9c69d2492b56d9dfe38573ed35f797a6132339ff49`, Metal `e261aa4cd3b1b6b9eeb4e1ba4f8699eb1f648a1c2032d9028d08a03b9918306f` at 2 micro-steps). The references below were taken with every rank on the machine's one device (`doc/gates/e2/ddp_gate_refs.sh`, `doc/gates/e2/win_gates.ps1 -Part ddpgate`; logs `doc/gates/e2/logs/*-ddp-gate.log`): at each configuration, 2 (and on Windows 4) ranks sharing the device gave the reference's final weights and every per-step hash.

| Device (key as in "GPU ranks") | Configuration | M = 2: reference, and W = 2 sharing the device | M = 4: reference, and W = 2 (Windows: and 4) |
|---|---|---|---|
| CPU, macOS aarch64 | gate (AdamW) | `9a24f95f50da08572824c0561890ea3213cc5076274b0cfaffe3bc6abba851d6` | `02ead67c808b59bde6852388e13d48a89c257b22a8080b95549bc780f2cf1a78` |
| | clip 1.0 | `382d20b6b7c970187cabb9909918c78cc7d4a23695de51d9d1c846fddacb99c0` | `5d4fa2a91c75d1803519b3674cd7acf231568b7e461ca71064fa0d1180f936b2` |
| | SGD with momentum | `9f5baec7e30edc422ce97941c1f600da9ed643cb70892b1821b1a74660417694` | `bb9775bd22f8edb3d99190f015fc25f74c3e54e36b257a4752257923795113f7` |
| | absent (12 steps) | `ad6123c3d8b63ef377a5a170e5ccb9fee8f0231149de96d1b541dbcc030c302d` | `ece0f094e6cda5b32da95963735ccd59533fcdda35dfc58321a197deec392a14` |
| | idle, decay after / torch (12 steps) | `f9bbdf2ac3769e419f5eee41b9479aa6edfff6cffb2b59b8a2ab4b2013b59292` / `2a8040b1464089c93726dda5bfacfb7c186ccc8ebec5dbd7b235e6c4deaf221b` | `f22326df4bbff1f8a33f8062c08f05c26d63e878f4b69213d8e700551e8deaf2` / `920cac4bb6c2f5a319549bb5fd33dc34a3dc01af85bae8980896aed83ebe98e4` |
| CPU, Windows x86_64 | gate (AdamW) | `3b94e85f76b93c1c250415836e41b3afaf15c70294d4ac01c17550f2becae08b` | `968508fe80c79c35340ee7227044ca2242038d6b7884858c979950c2680ead83` |
| CUDA, Windows, RTX 4060 Laptop GPU | gate (AdamW) | `3263b99095eded48b1d0cc9c69d2492b56d9dfe38573ed35f797a6132339ff49` | `4021208e74c5970b4ba0399cda524de9c497856635ab4a673066ebee72965d04` |
| | clip 1.0 | `d481ddb36d6a9c3f1a169e67d6eda1d1b6fa6a93edd3944a4d1edf30f30de873` | `561ebc59ea9cce995956425c60e59b15d3a1049428987ffb115ab00589ff93e3` |
| | SGD with momentum | `7650ce860e8db697d96e771ece5d20fca65c534d33353d6cc31fcb1ebf18d39c` | `d3de76efd88cc96efa82302de4f371ca789c0e6116c6960e2ab05a158d562c16` |
| | absent (12 steps) | `7e1eb912d531b3b7288387d3cc345af8152c3a26bc53db3494e719deb32b21ad` | `849193c50a05813d679e7525acbe6a442fda046a09f3d901ba0f3f69d3e5fb1d` |
| | idle, decay after / torch (12 steps) | `9a1a2a7a53e73db7edec9f80cc80462cab0032a3bca544dd4551726c1890b3d5` / `2a0ef10e35a0f1818b543b06c2563abf6b1c0cc57ea7c0b2fe2d92d0d3516b48` | `c5ff14a460d67746933ce01506e0ac4215dc8b8c0858285a8a649d158f22b69b` / `0ede0117a27e699212d2ab735a8ba66129924451cd3fcbf5d5d0edf3ac1eb2c7` |
| Metal, macOS, Apple GPU | gate (AdamW) | `e261aa4cd3b1b6b9eeb4e1ba4f8699eb1f648a1c2032d9028d08a03b9918306f` | `4d1adffd836bbd93cd25e3522f02c3f856feaf6c02b64ecd9afc0086767fdc21` |
| | clip 1.0 | `6f11db4d82361476e986921e29c8e9ab3708f41c6b6f4cef51536a41ff84ccdb` | `b43b6924c7fbe8d7ad8532ed0263a0771f81f68d3cdb6daeb72a5a3fdef84648` |
| | SGD with momentum | `fddba9d2e0cdc2d4b5a94144349bd7f41aea395b86cdeede8de490aaa04a877e` | `4ececb232bf4c0622feeb48ec454ec35630361fb9e9095643ed017296a40bdb9` |
| | absent (12 steps) | `0f687356977ebe24bcfb7112a71dca0202379db4398111867033d91ee30d9d2b` | `fadcaadeea04e4874251aa5d382207e702ddfdf2ac3893959f332bacc72765bf` |
| | idle, decay after / torch (12 steps) | `cb59dbc693e34af1099073de248e629dae6536df910d1140ae60f3c99220a87e` / `7880209a33b7201f67c5c121cc5bf0e0a14dafc3c57f8ad7ad735fe71c5dea04` | `e3cde77efef90afa267f8ec96d000c873517c41f2275dc7d422f3ddbb903b9fc` / `2a9045e9af861148fbdfa1d9b5e95e2e7842598c10ca540d8cee94ad373e3ede` |

Every entry is the reference's final weights (one process accumulating M micro-steps), and the runs with ranks sharing the device gave the same final weights and every per-step hash. The absent and idle runs take 12 steps (the library gates take 9 and 6), so their hashes are its own.

A resume (7 steps with checkpoints every 3, then a relaunch to step 12 from step 6) gave the uninterrupted weights and every per-step hash on the Mac CPU, on Metal and on CUDA; two launchers acting as nodes on the Mac (one process each, a shared secret file, checkpoints with node copies) gave the reference's weights, and node 1's checkpoint copy was node 0's byte for byte. One GPU per rank (`--ranks-per-device 1`) was tested after 0.6.0's release, on two L4 GPUs in one machine and across two machines with one L4 each (see "Two GPUs in one machine, and two machines").

## Sharded data parallelism (0.5.0)

Released in 0.5.0 (15 October 2026). Status: built; tested on one machine per device kind (the CPU reference backend on macOS aarch64 at world sizes 1 and 2 and on Windows x86_64 at 1, 2 and 4; CUDA with up to four processes sharing a single CUDA GPU on Windows; Metal with two processes sharing one Apple GPU on macOS; shown: byte-identical weights and per-step hashes to data parallelism at every stage and partition, with absent gradients, an idle weight under AdamW's decay and clipping, consolidated checkpoints byte-identical to data parallelism's, and a killed run resumed from a sharded checkpoint equal to the uninterrupted run); multi-GPU, multi-machine and datacenter runs untested.

`ShardedDataParallel` splits the optimizer's state, then the summed gradients, then the vars among the ranks, in the shape of ZeRO's three stages and of FSDP's full sharding, and every rank's weights stay byte for byte those of `data_parallel_step` at the same world size and micro-steps. It is opt-in: nothing else changes when it is not used.

| Item | What it does |
|---|---|
| `ShardedDataParallel::new(group, vars, groups, options, make)` | Shards `vars` (checked equal on every rank) and builds this rank's optimizer with `make(pieces, groups)` (`\|v, g\| Adam::new(cfg, g, v)`), as for whole vars. |
| `ShardOptions::new(stage)`, `with_partition` | `ShardStage::OptimizerState` (stage 1), `Gradients` (stage 2), `Parameters` (stage 3); `Partition::Flat` (default) or `ByVar`. Non-exhaustive, with a builder. |
| `step`, or `grads` then `apply` | One optimizer step, like `data_parallel_step`; `grads` returns this rank's pieces of the summed gradients (`ShardGrads`) for a caller that clips them first. |
| `ShardGrads::clip_grad_norm_per_seed`, `grad_norms_per_seed` | Clipping by the global norm per seed over the pieces of every rank: `clip_grad_norm_per_seed`'s norms, bit for bit. |
| `full_params` | The whole vars on every rank (gathered at stage 3: FSDP's full parameters). |
| `save_consolidated`, `restore_consolidated` | A checkpoint in `Checkpoint`'s format, the whole vars and optimizer state gathered: byte for byte data parallelism's at the same step; either kind of run resumes the other's. |
| `save_sharded`, `load_sharded_latest`, `restore_sharded` | Every rank writes and reads its own pieces (`dist-shard-checkpoint-1`). |
| `memory`, `record` | The bytes a rank holds (vars, summed gradient, optimizer state); the run's record with the sharding. |
| `ShardLayout`, `Piece` | Which elements of which var each rank holds. |

**Pieces.** Each rank holds a piece of every var: a range of its elements in row-major order, the ranges increasing with the rank. `Flat` cuts every var into W contiguous ranges (rank r takes `[n·r/W, n·(r+1)/W)`; FSDP's flat parameter, one per var), so the ranks hold within one element per var of an even share. `ByVar` gives every var whole to one rank: the vars from the largest, each to the least loaded rank (ties to the lowest rank), in the shape of torch's `ZeroRedundancyOptimizer`. A piece that is the whole var keeps the var's shape; any other piece is a rank-1 tensor of its elements. A var with no elements and a var that is not f32 are refused. Lazy rows, multiplier tensors and per-seed learning rates need a var's shape, so on a var cut into pieces each is refused by `ShardedDataParallel::new` with its own error ("lazy rows need its [S, R, C] shape", "its multiplier tensor broadcasts against its shape", "per-seed learning rates need its seed axis"; `flat_pieces_refuse_lazy_rows_multipliers_and_per_seed_rates`); `ByVar` keeps every var whole, and takes all three.

**Stages.**
1. `OptimizerState`: every rank keeps the whole vars; the gradients are all-reduced exactly as by `data_parallel_step`; each rank's optimizer steps its pieces, and the updated pieces are gathered back into every rank's vars.
2. `Gradients`: as 1, with a reduce-scatter in place of the all-reduce: each rank receives its own pieces of the summed gradient only.
3. `Parameters`: as 2, and between steps each rank keeps only its pieces of the vars; the whole vars are gathered at the start of a step for its micro-steps and dropped before the reduction. The whole model is gathered at once: there is no per-layer gather and free. Stage 3 therefore saves memory only between steps; during a step every rank holds the whole vars and its micro-steps' whole gradients, so its peak memory equals data parallelism's.

**Why the bits are data parallelism's** (deterministic mode, rules 1–9 above unchanged):
- **The reduce-scatter keeps the flat order.** It is the all-reduce's chain (rule 3): the global micro-steps folded by `GradSum` in flat (rank, micro-step) order along the ranks. Only its last step differs: the last rank sends each rank its own pieces instead of the whole total. Every element is the same fold of the same values in the same order; `Reduce::Mean` divides it by the same count (one shared routine). An all-reduce never meets a reduce-scatter: the pieces enter the layout both sides check.
- **The optimizers are elementwise.** Adam (both decay orders, the fused GPU update included) and SGD update every element on its own, so the same operations on a piece give each element the bits they give it in the whole var.
- **Gathering moves bits, never adds them.**
- **Absent and idle vars.** A var no micro-step on any rank used has no sum, so every piece of it is skipped by its optimizer, AdamW's decay included, as in one process. A var used by some micro-steps only is summed with zeros in place of its absent gradients (rule 1).
- **Clipping in single-process order.** `grad_norms_per_seed` sums each var's squares per seed in f64, element by element, then adds the vars' sums in their order. The sharded norm keeps both orders: each (var, seed) sum is one fold over the var's elements in row-major order, continued from rank to rank in rank order (the running sums pass along the ranks by broadcasts), and the sums are then added var by var. The factors multiply each piece's elements as they multiply the whole var's.
- **The key.** The group's key (rule 6) is unchanged: in the flat order the weights do not depend on the sharding, so sharded and data-parallel runs on the same key may be compared byte for byte, and either resumes the other's consolidated checkpoint. The record names the sharding (`ShardedDataParallel::record`: `… sharding=stage-3-parameters partition=flat`). Every sharded run's record names its stage and partition. A later sharding mode that changes the bits (any other order) gets its own key; none is built.

**Checkpoints.**
- **Consolidated** (`save_consolidated`): every rank gathers the whole vars and optimizer state, then `Checkpoint::save` runs as for data parallelism (every rank builds the bytes, rank 0 writes, every rank verifies). Its files are data parallelism's checkpoint at the same step, byte for byte, so `Checkpoint::restore` (data parallelism) and `restore_consolidated` (sharding, any stage or partition) both resume it. It needs the whole state on every rank for the moment of the save.
- **Sharded** (`save_sharded`): every rank writes its own pieces of the vars and of each optimizer set (`rank-<r>-params.json`, `rank-<r>-adam_m.json`, …) into one temporary folder; rank 0 writes `meta.json` (the step, the optimizer and its sets, the world size, the partition and the layout's key, the job and group keys, every cursor, each rank's state hash and each file's sha256), renames the folder into place, checks every rank's files against it and broadcasts its sha256; every rank compares it with its own meta file's; `latest` moves last. A failure on any rank fails the checkpoint on every rank. The folder must be one every rank can write (one machine, or a filesystem the ranks share).
- **Sharded resume** (`load_sharded_latest`, `restore_sharded`): every rank loads the same meta file (checked across ranks), its own files (checked against their sha256 and its state hash); another world size, partition or layout, job key or group key, or another kind of optimizer, is refused on every rank. Stages share the pieces, so a run may resume a sharded checkpoint written at another stage with the same partition.

**The gates** (deterministic mode; the gate's model, data and AdamW as in "The gate and its results"; every run a separate set of processes started by the launcher, per-step weight checks on every rank of the gathered vars). Logs: `doc/gates/e2/logs/`; scripts: `doc/gates/e2/run_gates.sh`, `doc/gates/e2/single_device_diff.sh`.

Each row: final weights and every per-step hash equal to data parallelism's run at the same world size and micro-steps (12 steps unless stated), at every listed stage and partition. Hashes are the final weights' sha256.

| Gate | Stages × partitions | macOS aarch64 CPU (W 1, 2) | Windows x86_64 CPU (W 1, 2, 4) | CUDA, RTX 4060 Laptop GPU (W 1, 2, 4 sharing it) | Metal, Apple GPU (W 1, 2 sharing it) |
|---|---|---|---|---|---|
| `gate_zero_every_stage_and_partition_equals_data_parallelism`, M 2 | 1, 2, 3 × flat, by-var (all six) | `9a24f95f50da08572824c0561890ea3213cc5076274b0cfaffe3bc6abba851d6` | `3b94e85f76b93c1c250415836e41b3afaf15c70294d4ac01c17550f2becae08b` | `3263b99095eded48b1d0cc9c69d2492b56d9dfe38573ed35f797a6132339ff49` | `e261aa4cd3b1b6b9eeb4e1ba4f8699eb1f648a1c2032d9028d08a03b9918306f` |
| same, M 4 | all six | `02ead67c808b59bde6852388e13d48a89c257b22a8080b95549bc780f2cf1a78` | `968508fe80c79c35340ee7227044ca2242038d6b7884858c979950c2680ead83` | `4021208e74c5970b4ba0399cda524de9c497856635ab4a673066ebee72965d04` | `4d1adffd836bbd93cd25e3522f02c3f856feaf6c02b64ecd9afc0086767fdc21` |
| `gate_zero_absent_gradients_and_an_idle_weight_equal_one_process`: absent, 9 steps, M 2 / M 4 | all six (GPU: 1 flat, 2 by-var, 3 flat) | `7c61a7994ceb0d787ef5ca45a5fd604075a5d3f21c1cf84add0cd59ee590d67c` / `0ad5e705faca0758a2d0fd8dc06a82106a99c5c3f4d1962417e0e2f27f79d3e7` | `c0e3c2da36b137d5dc2ec6c2061e926d05415484e29dbac919496b397637133b` / `6e704dcc8af711e2318f4e7561272616951d390a75c775ea5d0cf448d1538ca1` | `ed14a6550cd5773878d04a6516ab7f3c5a9ab9df6768db24c6f7f8d0c907aa1c` / `5bae8705a1f990ecfdbae8f8f19619b914347bdd753de2d0fb092cfc29ff6f60` | `055998bd589d27e9f2755f18d42c47564edb6b530010d30ed5735011ae4ddc76` / `0ef5c715415531e36d293ef4e38a109ae1026c3e3ec99a631b5f2a162e0fcf41` |
| idle weight, decay after / torch, M 4, 6 steps (its bits untouched) | as above | `2a30f578eadf26ff812e297ddea86d0228457322d3e839197e015ce0be08638e` / `4decceea205f0f8a6581050428afc6932b0aca0390ee476137863d3df938db53` | `31c5d0b3bdceaa6d274f5816a0cd79374e4013cf2a34121049661038dd3e973d` / `db9186e5296c2eaa33116a07b378b33428f69aea16e628158ad39f27ec397d7e` | `54f777084d76890aa834b87be8c76a77a5861386c125d901935e3e0fdcf7bc2f` / `136006461c91ed08f4845a58562cc4c5a433fb40507c805f593c6d8f76d33da0` | `91867037aef3b6445fb2d246a188a2b76ead652ef5b4b8586911a7f76b3ed38a` / `f51d05efa6242d0828b2710d76ab5ab39b6e341345b70e6df770b703356c7367` |
| `gate_zero_clipping_equals_data_parallelism`, clip 1.0 (engaged on 12 of 12 steps), M 2 / M 4 | as above | `382d20b6b7c970187cabb9909918c78cc7d4a23695de51d9d1c846fddacb99c0` / `5d4fa2a91c75d1803519b3674cd7acf231568b7e461ca71064fa0d1180f936b2` | `aa1b37445adbfc1e269380a7a84ae4cb0b8a310558ca89c0dc723f714fe215c3` / `dae9d10a3171b13c08f938804a539f946fc93f88bd54895c403bcf2b7c452034` | `d481ddb36d6a9c3f1a169e67d6eda1d1b6fa6a93edd3944a4d1edf30f30de873` / `561ebc59ea9cce995956425c60e59b15d3a1049428987ffb115ab00589ff93e3` | `6f11db4d82361476e986921e29c8e9ab3708f41c6b6f4cef51536a41ff84ccdb` / `b43b6924c7fbe8d7ad8532ed0263a0771f81f68d3cdb6daeb72a5a3fdef84648` |

The absent and idle references are one process that puts zeros in place of the absent gradients and adds them with the engine's own tensor additions (no `GradSum`), as in 0.3.0's gates. The macOS CPU, CUDA and Metal hashes and the Windows clipping hashes are the ones 0.3.0 and 0.4.0 recorded; the other Windows CPU hashes were not recorded before (data parallelism gives them in the same logs).

`gate_zero_consolidated_checkpoints_are_data_parallelisms`: checkpoints every 3 steps (W 2, M 2) or every 4 (W 4, M 4) of 12; every file of every checkpoint (`vars.json`, the optimizer files, `meta.json`, `latest`) byte-identical to data parallelism's: Adam at W 2 (17 files) and W 4 (13 files, not run on the Mac), SGD with momentum at W 2 (13 files); on the CPU at all six stages and partitions, on CUDA and Metal at stage 1 flat, stage 2 by-var and stage 3 flat. `consolidated_and_data_parallel_checkpoints_resume_into_each_other` (2 ranks): a data-parallel checkpoint resumed by a sharded run, and a sharded run's consolidated checkpoint resumed by data parallelism, both reach the vars of 4 straight data-parallel steps.

`gate_zero_kill_and_sharded_resume_equals_the_uninterrupted_run`: sharded checkpoints; a rank paused and killed by the launcher; the job stops; the relaunch resumes from the last sharded checkpoint. Final weights and every per-step hash equal the uninterrupted sharded run's, which equals data parallelism's. Resuming the same checkpoint with one process, or with the other partition, is refused.

| Case | Killed | Resumed from | macOS CPU | Windows CPU | CUDA |
|---|---|---|---|---|---|
| stage 3 flat, AdamW, W 2, M 2 | rank 1 at step 7 | step 6 | `9a24f95f50da08572824c0561890ea3213cc5076274b0cfaffe3bc6abba851d6` | `3b94e85f76b93c1c250415836e41b3afaf15c70294d4ac01c17550f2becae08b` | `3263b99095eded48b1d0cc9c69d2492b56d9dfe38573ed35f797a6132339ff49` |
| stage 2 by-var, SGD with momentum, W 2, M 2 | rank 1 at step 7 | step 6 | `9f5baec7e30edc422ce97941c1f600da9ed643cb70892b1821b1a74660417694` | `2cff4c2a35d5ba1a70e9eed9b1f941174336d33d49675bc62dd9d1134c1454b2` | `7650ce860e8db697d96e771ece5d20fca65c534d33353d6cc31fcb1ebf18d39c` |
| stage 1 flat, AdamW, W 4, M 4 | rank 2 at step 5 | step 4 | not run (2 processes at most) | `968508fe80c79c35340ee7227044ca2242038d6b7884858c979950c2680ead83` | `4021208e74c5970b4ba0399cda524de9c497856635ab4a673066ebee72965d04` |
| stage 3 by-var, AdamW, W 4, M 4 | rank 2 at step 5 | step 4 | not run | `968508fe80c79c35340ee7227044ca2242038d6b7884858c979950c2680ead83` | not run (CPU only) |

Metal ran the first two cases (`e261aa4cd3b1b6b9eeb4e1ba4f8699eb1f648a1c2032d9028d08a03b9918306f`, `fddba9d2e0cdc2d4b5a94144349bd7f41aea395b86cdeede8de490aaa04a877e`).

`a_sharded_checkpoint_round_trips_byte_for_byte_and_resumes_exactly` (2 ranks; stage 3 flat under AdamW, stage 1 by-var under SGD with momentum): a sharded checkpoint loaded into a fresh run and saved again gives every file byte for byte (8 and 6 files); the restored run continued 2 steps equals 4 straight steps. `the_reduce_scatter_is_the_all_reduce_cut_into_pieces` (W 1, 2, 4; chunks of 5 and 2²⁰; absent gradients; flat and by-var) and `sharded_clipping_equals_clipping_the_whole_sum` (W 1, 2, 3, 4; norms and clipped pieces bit for bit, binding and non-binding bounds) check the two collectives at the gradient level; `sharded_steps_equal_data_parallel_steps_in_one_process` checks the vars and both Adam moments with distribution off.

**Single-device and data-parallel behaviour is unchanged.** With sharding unused, every 0.3.0 and 0.4.0 gate above gives its recorded hashes (the logs in `doc/gates/e2/logs/` print them). The six single-device training configurations (a decoder, a mixture-of-experts decoder and a decoder with dropout, each under AdamW and under SGD with momentum, 60 steps, 3 models on the seed axis; final vars, logits on a fixed batch and every per-step loss) and the copy-task example print the same bytes built from 0.4.0 and from this change (`doc/gates/e2/single_device_diff.sh`, `doc/gates/e2/single_device_probe.rs`), on the CPU reference backend (macOS aarch64, Windows x86_64), on CUDA and on Metal. Outside `nn::dist` nothing changes; inside it, `GradSum::finish` now calls the scaling routine it shares with the reduce-scatter (the same operations), and the all-reduce's chain gained the reduce-scatter's last step behind a parameter.

**Where it ran.** The Mac was shared with a long benchmark: its runs used at most 2 processes at once, at background priority, so world size 4 ran on Windows only (CPU and CUDA). Windows ran every gate with no skips, under rustc 1.94.1.

**Memory per rank** (`sharding_divides_the_optimizer_state_gradients_and_vars`; the gate's model, 43,840 f32 elements = 175,360 bytes of vars; Adam's state is two moments, 350,720 bytes; after one step; bytes per rank, rank 0 first). Data parallelism holds, on every rank, 175,360 bytes of vars, 175,360 of summed gradient and 350,720 of optimizer state.

| Stage, partition | W 2: vars / summed gradient / optimizer state | W 4: vars / summed gradient / optimizer state |
|---|---|---|
| 1, flat | 175,360 / 175,360 / 175,360 each | 175,360 / 175,360 / 87,680 each |
| 2, flat | 175,360 / 87,680 / 175,360 each | 175,360 / 43,840 / 87,680 each |
| 3, flat | 87,680 / 87,680 / 175,360 each | 43,840 / 43,840 / 87,680 each |
| 1, by-var | 175,360 / 175,360 / 176,128 and 174,592 | 175,360 / 175,360 / 90,112, 90,112, 86,016, 84,480 |
| 2, by-var | 175,360 / 88,064 and 87,296 / 176,128 and 174,592 | 175,360 / 45,056, 45,056, 43,008, 42,240 / 90,112, 90,112, 86,016, 84,480 |
| 3, by-var | 88,064 and 87,296 / the same / 176,128 and 174,592 | 45,056, 45,056, 43,008, 42,240 / the same / 90,112, 90,112, 86,016, 84,480 |

With flat pieces each rank's optimizer state is exactly 1/W of data parallelism's (0.500 at W 2, 0.250 at W 4); by var, the largest rank holds 0.502 and 0.257 (the vars are whole, so the split is only as even as their sizes). The rows sum to data parallelism's bytes: the state is split, not copied. Measured from the tensors each rank holds (`ShardedDataParallel::memory`), in `sharding_divides_the_optimizer_state_gradients_and_vars` on Windows (all world sizes) and on the Mac (W 1 and 2).

Not counted: each micro-step's own gradients, which are whole on every rank at every stage (a rank after the first keeps its k micro-steps' gradients until the running sum arrives, as in data parallelism), the whole sum the last rank holds during the reduction, and the whole vars a stage-3 step gathers.

**Refusals:** a sharded checkpoint resumed at another world size or with another partition (refused on every rank, in the gates and in `a_sharded_checkpoint_round_trips_byte_for_byte_and_resumes_exactly`), a changed byte in another rank's file (both ranks refuse), ranks that start from different vars, a var that is not f32, per-seed rates on vars cut into pieces (`sharding_refuses_what_its_pieces_cannot_hold`).

**Untested and not built.** Sharding on more than two GPUs, or across machines (two L4 GPUs in one machine were tested after 0.6.0's release, stages 1–3 with flat pieces: see "Two GPUs in one machine, and two machines"); a sharded checkpoint on a filesystem shared by several machines; per-layer gathering (FSDP's units), prefetching and overlap; offloading to the host; mixed precision; a sharded checkpoint resumed at another world size (refused); sharding in an order other than the flat one.

## Fast mode (released in 0.6.0)

Status: built; tested on one machine per device kind (the CPU reference backend on macOS aarch64 at world size 2 and on Windows x86_64 at 2 and 4; Metal with two processes sharing one Apple GPU on macOS; CUDA with up to four processes sharing a single CUDA GPU on Windows; shown: the ranks hold the same weights after every step, the per-step loss stays within the tolerance below of the deterministic mode's over 12 steps, a job mixing modes or buckets is refused before step 0, and with the mode off the 0.5.0 results are unchanged); multi-GPU, multi-machine and datacenter runs untested. Its results are fast mode, not reproducible bit for bit: they are never pooled with, or compared byte for byte against, the deterministic mode's.

**What it does** (`GroupOptions::new(…).with_mode(DistMode::Fast(FastOptions::new()))`; `DistMode::fast()` for the defaults):
- **Per-rank sums.** Each rank adds its own micro-steps in order first; then the ranks' sums are reduced. The bits depend on the world size.
- **Buckets.** The summed gradients go in buckets of at most `FastOptions::bucket_bytes` (25 MiB by default, torch DDP's `bucket_cap_mb`; a var larger than that is a bucket of its own), filled with the vars in reverse order, the order in which a backward pass usually finishes them. Every rank cuts the same buckets and reduces them in the same order.
- **A ring.** Each bucket is reduced by a ring over the existing TCP mesh: a reduce-scatter in W − 1 steps (each rank passes a segment's running sum to the next rank, which adds its own), then an all-gather in W − 1 steps. Each element of the sum is added once, by its segment's chain, and then copied, so every rank holds the same bits and the vars stay in sync (`check_in_sync` still checks it every step in the gates); the order of the additions depends on W and on where the segments fall, and is not the flat rank order. A sending thread per rank writes to the next rank while the rank reads from the previous one.
- **Sharded training** (`ShardedDataParallel` on a fast group): stage 1 takes the ring all-reduce; stages 2 and 3 take the ring reduce-scatter alone, its segments being the ranks' pieces, so each rank receives only its own. The gathers of the vars and the clipping norms are the deterministic mode's (they move bits, or keep their order).
- **Overlap with the backward pass.** With `FastOptions::overlap` (on by default), the backward pass of each rank's last micro-step names each var as soon as its gradient is final (its last consumer has run), and a bucket goes to a communication thread as soon as all its vars are final; the ring reduces it while the backward pass continues. Earlier micro-steps are summed locally, so there is nothing to send during their backward passes (torch DDP's `no_sync` does the same). On a GPU each gradient is copied to the host as it is named. Overlap changes when a bucket is reduced, not what is added: overlap on and off give the same sum (`the_fast_all_reduce_is_the_sum_of_every_ranks_micro_steps`). The backward pass's own traversal and additions are unchanged; the notification only reads the gradients.
- **The fast CPU backend** (`Cpu(Fast)`) is a device of the fast mode (`device=cpu-fast` in the key); the deterministic mode still refuses it. Not used in the gates.

**Keys, records and refusals.**
- The key: `mode=fast order=ring world=<W> bucket=<bytes> overlap=on|off backend=tcp platform=<platform key>` (with the device's part on a GPU). It never equals a deterministic key, and it carries the world size and the buckets, on which the bits depend.
- The record (`ProcessGroup::record`) is the key, the world size, the authentication and the weight checks, followed by "(fast mode, not reproducible bit for bit)"; a sharded run's record then names its stage and partition.
- A fast rank's hello carries its mode (`mode=fast order=ring bucket=… overlap=…`, after the platform key; a deterministic rank's hello is 0.5.0's). Rank 0 refuses a job whose ranks ask for different modes, or for the fast mode with other buckets or overlap, before step 0, with the reason on every rank ("a mixed-mode job"). A 0.5.0 rank 0 refuses a fast rank as a mixed platform.
- A checkpoint holds its group's key, so a fast run's checkpoint is refused by a deterministic resume and the other way round.
- `ProcessGroup::require_deterministic(what)` is Ok in the deterministic mode and refuses the fast mode, for any caller that needs results it may compare byte for byte or pool.

**Not built.** NCCL and its reductions (later, after a two-machine gate); a tree all-reduce; reduction in arrival order (the ring's order is fixed by W and the buckets; on one machine and key, a repeated fast run gave the same per-step hashes, an observation that the mode does not promise); rebuilding the buckets from the order a first step finishes the vars; compression; overlap during the earlier micro-steps or with the optimizer step.

**The gates** (`doc/gates/e4/`; logs in `doc/gates/e4/logs/`). The gate's model and data (12 steps, M = 4 global micro-steps, AdamW), each rank started by the launcher, all ranks sharing the machine's one device, per-step weight checks on every rank; the deterministic reference is one process accumulating the 4 micro-steps. The loss compared is each step's mean over the 4 global micro-steps (`ddp_gate --losses`).

| Device (key as in "GPU ranks") | W | Fast-mode final weights (buckets of 25 MiB) | Largest per-step loss difference against the deterministic run, 12 steps | Over 200 steps |
|---|---|---|---|---|
| CPU, macOS aarch64 | 2 | `66f8c67e923a52a7a04f06240a4327021ee986dd5343b065c43bb527efe408cc` | 4.77e-7 (relative 1.16e-7) | 4.39e-2 (at step 189) |
| Metal, Apple GPU | 2 | `e63c5f1097171dcd27de0b813f47a3c9645eace18849d4d7f5444ea8b6caa755` | 1.79e-7 (relative 6.0e-8) | 6.77e-2 (at step 200) |
| CPU, Windows x86_64 | 2 | `50217b73b282a8d5ac5e3d353b027a8daa69934dc3ecf5efb1e408350ebdad91` | 4.17e-7 (relative 1.47e-7) | 1.23e-2 (at step 195) |
| | 4 | `e556f83bbdd8eb2b88010e08ea87a72bdff612affd73e9c03560fa8bb39a97bc` | 2.39e-7 (relative 8.4e-8) | 5.49e-3 (at step 200) |
| CUDA, Windows, RTX 4060 Laptop GPU | 2 | `c341e1ef9a374e865a9ec9a90d7a375916767d6dc9d39e0ff240d268a1597229` | 3.57e-7 (relative 7.2e-8) | 1.58e-2 (at step 200) |
| | 4 | `51014fb72db074d9107db8b3803c0f62ea182bd32943533270c75af4f735e559` | 2.98e-7 (relative 7.2e-8) | 1.73e-2 (at step 195) |

- **The tolerance, set with the measurement:** over the gate's 12 steps, the fast mode's per-step mean loss was within 5e-7 absolute (2e-7 relative) of the deterministic mode's on every configuration run, with buckets of 25 MiB (one bucket) and of 16 KiB (several), overlap on and off; 5e-7 was set from this measurement, not fixed in advance. It is a tolerance, not an identity.
- **Over 200 steps** the two runs drift apart as their rounding differences compound through training: on the Mac CPU the difference first reaches 1e-6 at step 54 and 4.4e-2 at step 189; both runs train (mean loss over steps 191–200: deterministic 0.0661, fast 0.0625, from 6.55 at step 1). Such a drift is expected of any change of summation order and is why the mode is never compared byte for byte.
- **At M 2, W 2** each rank has one micro-step: the ring adds the two micro-steps' gradients once, and f32 addition commutes, so the fast mode gave the deterministic bits there (Mac CPU, `9a24f95f50da08572824c0561890ea3213cc5076274b0cfaffe3bc6abba851d6`). At M 4, W 2 it adds `(a + b) + (c + d)` where the deterministic mode adds `((a + b) + c) + d`, and the bits differ. At W 4 the bits also depend on the buckets (on Windows, 16 KiB buckets gave other bits than 25 MiB ones); at W 2 they did not (a two-rank ring adds each element's two sums once).
- **Sync and checkpoints:** every per-step weight check passed on every rank in every fast run; a fast run checkpointed at step 6 and resumed gave the uninterrupted fast run's weights (an observation); a deterministic resume of the fast checkpoint was refused (Mac CPU and Metal).
- **Refusals:** two launchers acting as nodes, one rank asking for the fast mode and one for the deterministic mode, and two fast ranks asking for different buckets: refused on both ranks before step 0 (Mac CPU and Metal).
- **Library tests** (`nn::dist::tests::fast_tests`, ranks as threads): the ring sum within f32 rounding of the exact sum at W 1, 2, 3 and 4, with buckets smaller than a var, absent vars kept absent, the same bits on every rank; the mean divides by every rank's micro-steps; each rank's sharded pieces at every stage and partition (W 2 and 3); the backward pass names every var once, with `backward`'s bits, the last vars first; the gate's model at W 2 for 6 steps within 4.8e-7 in loss and 5.4e-6 in weight of the deterministic run, data-parallel and sharded (stage 3 flat, stage 2 by var); the keys, records and refusals.

**Wall time** (`doc/gates/e4/timing.sh`, log `mac-cpu-timing.log`; a measurement on one machine, not a statement about other hardware, networks or engines). Apple silicon, model MacBookPro18,1 (8 performance and 2 efficiency cores, 16 GiB), macOS 27.0.1, the CPU reference backend, localhost TCP; every process at background priority (`taskpolicy -b`, which macOS runs on the efficiency cores), on a machine shared with other work (load average 4.4 to 9.0 during the runs). A decoder of width 256, 4 blocks, MLP 1024 (4,206,848 parameters per model, 2 models on the seed axis: 33.7 MB of f32 gradients a step), 4 global micro-steps of 2 items, 10 steps; the training loop's wall time on rank 0; 5 runs per configuration, in rotation.

| Configuration | Key (the platform part as in the deterministic key) | Median, 10 steps | Runs (seconds) |
|---|---|---|---|
| 1 process, 4 accumulation micro-steps | `mode=deterministic order=flat-rank backend=tcp platform=macos-aarch64-neon-rustc_1.94.1-matrixmultiply-0.3.11 libm=<sha256>` | 16.05 s | 16.36, 16.05, 16.61, 14.80, 15.53 |
| deterministic, W 2 | the same | 18.85 s | 18.85, 19.22, 23.92, 17.92, 18.05 |
| fast, W 2, buckets of 25 MiB, overlap on | `mode=fast order=ring world=2 bucket=26214400 overlap=on backend=tcp platform=…` | 18.34 s | 18.34, 19.89, 17.34, 18.93, 17.89 |
| fast, W 2, buckets of 25 MiB, overlap off | `… bucket=26214400 overlap=off …` | 18.83 s | 20.52, 18.83, 18.00, 19.10, 17.64 |
| fast, W 2, buckets of 1 MiB, overlap on | `… bucket=1048576 overlap=on …` | 18.38 s | 19.57, 34.67, 18.38, 17.97, 17.57 |

On this machine and load the fast mode's medians are within 3% of the deterministic mode's, inside the runs' spread: the measurement shows no difference between the modes here, either way. Both two-rank runs took longer than one process: the two ranks ran on the two efficiency cores of a loaded machine. These numbers say nothing about other machines, real networks or GPUs, where the share of communication differs.

**With the mode off, nothing changes** (gate 1; `doc/gates/e4/run_gates.sh`, `ddp_gate_refs.sh`, `single_device_diff.sh`): the `nn::dist` tests pass on the Mac (59, the runs needing more than 2 processes at once skipped); `examples/ddp_gate.rs`'s 26 reference and two-rank results (the gate, clipping, SGD, absent and idle configurations at M 2 and 4, a resume, two launchers as nodes) print 0.5.0's final weights line for line on the CPU (`9a24f95f50da08572824c0561890ea3213cc5076274b0cfaffe3bc6abba851d6` at M 2, `02ead67c808b59bde6852388e13d48a89c257b22a8080b95549bc780f2cf1a78` at M 4) and on Metal; the six single-device configurations and the copy-task example print the same bytes built from 0.5.0 and from this change, on the CPU and on Metal; the sharding gates (every stage and partition equal to data parallelism) pass among the `nn::dist` tests. The deterministic gates were not rerun on Windows, nor at W 4 on the Mac.

## Mixed precision (released in 0.6.0)

See `doc/mixed_precision.md`. A group made while the thread's matmul precision is `Half` records it, and its key's device part names the reduced-precision product, in either mode; every data-parallel or sharded step checks that the thread multiplies at the group's precision. The mixed steps scale the loss by a static power of two, scale the summed gradients (or a rank's pieces) back, and skip the optimizer step on every rank when a gradient on any rank is not finite. Within a mixed key, two-rank data parallelism and two-rank sharded training (stage 2, flat) gave one process's weights bit for bit on CUDA and Metal (the gate's model, 12 steps); other world sizes, stages and partitions under mixed precision are untested. Never compared with f32.

## Two GPUs in one machine, and two machines (after 0.6.0's release)

Tested on rented cloud machines, with each setup's key below:
- 2 L4 GPUs in one machine (driver 580.126.09): data-parallel with one rank per GPU (world size 2) and two ranks per GPU (world size 4), and sharded (stages 1–3, world size 2), equal to one GPU with accumulation;
- 2 machines with one L4 each over a private network (driver 595.91.07): data-parallel with one GPU each (world size 2) and on the CPU (world sizes 2 and 4), equal to one machine.

Deterministic path only. Untested: more than 2 GPUs or 2 machines, other GPU models and drivers, NCCL, and the fast mode, half-precision products and mixed precision on these machines.

These runs used the 0.6.0 release unchanged, with every 0.6.0 option off: the fast mode, the half-precision products and mixed precision were not tested on these machines, and their own tests were skipped by name (`fast_tests`, `mixed_tests`, `half_gemm`). Sharding was tested on two GPUs in one machine only, not across machines. Each result holds for its own key below; the two setups' keys differ (driver, CPU features), and no result is pooled across them.

### Two L4 GPUs in one machine

**The key.** NVIDIA L4 × 2 (sm89, 24 GB), NVIDIA driver 580.126.09 (CUDA driver API 13.0), CUDA toolkit 12.8 (NVRTC 12.8.93, cuBLAS 12.8.4.1), Ubuntu 24.04.3, an AMD EPYC 7542 host (avx, avx2, fma), rustc 1.94.1. Platform key `linux-x86_64-avx+avx2+fma-rustc_1.94.1-matrixmultiply-0.3.11` with its maths-library fingerprint; device key `cuda-NVIDIA_L4-sm89-driver13000-nvrtc12.8-nofastmath-fmadoff-kernels-` and the first 16 characters of the CUDA kernel source's sha256 (aa58ccd1173439a053ba1ddd91326d3c6f33804a072a7a168c84a038f099e60c). The ranks reach each other on the machine's loopback address.

**Shown** (the gate's model, 12 steps; "equal" means every per-step weight hash on every rank and the final weights byte-identical):
- the engine's tests, CPU and CUDA: all passed (the CUDA tests with every rank on `Cuda(0)`, as on one GPU);
- the probe twice on each GPU, byte-identical; the gate's training on `Cuda(0)` and on `Cuda(1)`, equal;
- 2 ranks, one per GPU (`--ranks-per-device 1`), against one process on `Cuda(0)` with 2 micro-steps: equal, in each configuration (the gate's AdamW with dropout keyed by the global micro-step; absent gradients with an idle weight under decay; clipping at 1.0; SGD with momentum);
- 4 ranks, two per GPU, against one process with 4 micro-steps: equal;
- 2 ranks on 2 GPUs with checkpoints every 3 steps, rank 1 killed before step 7, the relaunch resumed from step 6: equal to the uninterrupted run;
- sharded at stages 1, 2 and 3 (flat pieces), 2 ranks on 2 GPUs: equal to the two-rank data-parallel run;
- CPU data parallelism at 1, 2 and 4 processes: one final-weights hash.

### Two machines with one L4 each

**The key.** On each machine: NVIDIA L4 × 1 (sm89), NVIDIA driver 595.91.07 (CUDA driver API 13.2), the same toolkit, Ubuntu 24.04.3, an AMD EPYC 9554 host (avx, avx2, fma, avx512f), rustc 1.94.1. Both machines had the same platform key (`linux-x86_64-avx+avx2+fma+avx512f-rustc_1.94.1-matrixmultiply-0.3.11` with its maths-library fingerprint) and the same device key (`cuda-NVIDIA_L4-sm89-driver13020-nvrtc12.8-nofastmath-fmadoff-kernels-`, as above). The machines were in one data centre, on a private network. The rendezvous and the data listeners were bound to the private address, a connection to the first machine's public address at the rendezvous port was refused, and every rank authenticated with HMAC-SHA256 over a per-job secret.

**Shown:**
- on each machine, twice identical: CPU `ddp_copy_task` with 4 micro-steps, and the GPU gate at one GPU with 2 micro-steps in each configuration;
- CPU data parallelism across the machines, 1 process per machine (world size 2) and 2 per machine (world size 4): every rank's final weights equal to one process with accumulation; the group record of world size 2 says `auth=hmac-sha256`;
- the GPU gate across the machines, 1 GPU per machine (world size 2), against one GPU with 2 micro-steps on one machine: equal, in each configuration;
- rank 1 (the second machine) killed before step 7, the relaunch resumed from step 6: equal to the uninterrupted run on both ranks; the second machine's checkpoint copy byte-identical to the first's;
- refused on every rank before step 0: a rank with another job secret, a rank with none, and a CPU rank in a GPU job;
- every rank checked every rank's data-shard hash against its own computation.

**Not shown.** No step time is given here: these runs measured whether the result changes across GPUs and machines, not speed, and two GPUs or two machines say nothing about larger configurations.

## Single-device behaviour is unchanged

- No existing file changes behaviour. The changes outside `nn::dist`: the module line in `nn/mod.rs`, a new error variant `NnError::Dist` (kind "dist"), two examples, this note and a README bullet.
- All 88 existing library tests and the README doctests pass.
- Six training configurations (decoder, MoE decoder, decoder with dropout; each with AdamW and SGD with momentum; 60 steps, 3 models on the seed axis) give byte-identical final vars and logits built against 0.2.0 and against this branch; the copy-task example's output is identical.
- `data_parallel_step` with the single-process group and one micro-step equals `nn::train_step` bit for bit (vars and Adam moments, `one_process_with_one_micro_step_equals_the_plain_trainer`).
- **GPU ranks (0.4.0):** the same six configurations give byte-identical final vars and logits built against 0.3.0 and against the GPU-ranks change (0.3.0 built and run now as the reference, on the same machine and device), on the CPU reference backend (macOS aarch64 and Windows x86_64), on CUDA (`Cuda(0)`, Windows) and on Metal (macOS). The CPU gates above give the same hashes as 0.3.0 (on Windows every one of them; on the Mac those that run in at most 2 processes at once, see the machine note in the change log), and the CPU key is unchanged. The other changes outside `nn::dist`: `tensor::gpu_platform_key`, the CUDA backend on `Cuda(i)` (one device per process), NVML loaded at run time, `Sgd::set_buffers`; no kernel source changes.

## For the release notes

- 0.6.0 (fast mode): `DistMode` (non-exhaustive; `Deterministic` the default, `Fast`), `FastOptions` (non-exhaustive, with a builder), the public field `GroupOptions::mode` and `with_mode`, `ProcessGroup::mode`, `is_fast` and `require_deterministic`; two new message kinds on the wire (the ring's), exchanged only between fast-mode ranks; the hello of a fast-mode rank carries its mode, the deterministic hello is unchanged ; the wire protocol is 3 (a 0.5.0 rank, `wire=2`, is refused before step 0); `ddp_gate` gains `--fast`, `--bucket-kb`, `--no-overlap`, `--losses`, `--width`, `--blocks`, `--mlp` and prints its loop's wall time

- adds the variant NnError::Dist
- public API additions (`nn::dist`, among them `GroupOptions` with public fields `platform` and `secret`, `JobSecret`, `platform_key`, `LaunchConfig::secret`): a public API change, so the release carrying them needs a minor version bump
- 0.5.0 (versions at the rendezvous): `wire_version`, `WIRE_PROTOCOL`; the hello's platform field carries the versions after the platform key, so 0.4.0 and 0.5.0 ranks never form a job
- 0.5.0 (sharding): `ShardedDataParallel`, `ShardOptions` (non-exhaustive, with a builder), `ShardStage`, `Partition`, `ShardGrads`, `ShardLayout`, `Piece`, `ShardMemory`, `ShardCheckpoint`, `ShardCheckpointMeta`, `ShardRank`, `SHARD_CHECKPOINT_FORMAT`; a new message kind on the wire (the reduce-scatter's result; data parallelism's messages are unchanged); the example `ddp_gate`
- 0.4.0 (GPU ranks): `GroupOptions` gains the public field `device` and is `#[non_exhaustive]` with a builder (`new`, `with_*`; struct literals no longer compile outside the crate); `DeviceKind`, `DistEnv::device`, `device_key`, `ProcessGroup::device` and `ProcessGroup::platform`, `data_parallel_grads`, `CheckpointOptimizer`, `Sgd::set_buffers`, `tensor::gpu_platform_key`; `Checkpoint::save` and `restore` take any `CheckpointOptimizer` (Adam or SGD), `Checkpoint` holds `optimizer_state` in place of `adam_m` and `adam_v`, and `CheckpointMeta` gains `optimizer`; NVML is loaded at run time and `device_key` is an error without it; `Cuda(i)` with i > 0 is accepted (one device per process, untested) where 0.3.0 refused it; a resume under another group key, or into another kind of optimizer, is refused

## Known limits and next steps

- **f32 gradients only.** f64, f16 and bf16 gradients are refused by `GradSum`. bf16 with f32 master weights comes with mixed precision.
- **Tested on two GPUs and on two machines, no more.** GPU gradients are copied to the host for the sum (correct, not fast). After 0.6.0's release the deterministic mode passed the gates of 1 node × 2 GPUs and 2 nodes × 1 GPU against 1 GPU × 2 micro-steps, each on its own key (see "Two GPUs in one machine, and two machines"); more GPUs, more machines and other GPU models are untested.
- **The OS maths library.** softmax, log_softmax and `powf` call the platform's `exp`, `log` and `pow`, whose last bits may differ between operating systems or library versions. The gate runs on one machine, so it is unaffected. Between machines, the rendezvous refuses ranks whose maths library fingerprint differs (the fingerprint samples the functions on fixed inputs; it can miss a difference elsewhere, which `check_in_sync` then reports as an error rather than letting the ranks drift). Mixed-platform jobs need the engine's own implementations of these functions.
- **Throughput (deterministic mode).** The running sum passes the ranks one after another (latency rises with W) and the last rank sends the total to each rank (its traffic rises with W). Both keep the order; a tree or ring broadcast of the total, bucketing and overlap with the backward pass in this mode are later work and must not change the order of the additions. The fast mode has buckets, a ring and overlap, under its own key.
- **Memory.** A rank after the first keeps its k micro-step gradients until the running sum arrives (k copies of the gradients; with one micro-step per rank, none extra).
- **Checkpoints.** Rank 0 is the single writer; node copies are optional and checked. The optimizer states saved are Adam's and SGD's (SGD's since 0.4.0); another optimizer cannot be checkpointed (it does not implement `CheckpointOptimizer`).
- **Not yet built:** elastic restart, generic callbacks (batch seen, checkpoint reached), NCCL (later, as an opt-in under its own key, never pooled with the deterministic mode; the two-machine gate it waited for has passed). The fast mode and mixed precision are built and released in 0.6.0 (see "Fast mode" and `doc/mixed_precision.md`).
