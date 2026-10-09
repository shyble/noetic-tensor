# Distributed training: deterministic data parallelism (first step)

9 October 2026, on top of the 0.2.0 release. CPU only; multi-GPU and multi-machine runs are later steps.

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
| `LaunchConfig`, `spawn`, `run`, `Job` | A torchrun-style launcher: N processes per node, rendezvous variables set, a log per rank; the first failure stops the job. |
| `examples/launch.rs` | The launcher as a command: `launch --nproc-per-node N [--nnodes M --node-rank I --master-addr A --master-port P] [--log-dir D] -- <command> [args]`. |
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

1. **One order for the sum.** A step's gradient is the left fold over its global micro-steps, `((g₀ + g₁) + g₂) + …`, elementwise in f32. Rank r's i-th of k micro-steps is global micro-step `r·k + i`. The fold starts from the first contribution (never from zeros, which would turn −0 into +0); a var without a gradient in a micro-step adds nothing; a var no micro-step touched stays `None` (the optimizer skips it, as it does today).
2. **One routine.** `GradSum` is the only place gradients are added, for local accumulation and for the all-reduce. `Reduce::Mean` divides by the number of global micro-steps as `Tensor::div_scalar` does, and not at all when there is one (so a single micro-step passes through bit for bit).
3. **Rank order across the network.** The all-reduce passes the running sum along the ranks: rank 0 sends its sum to rank 1, which folds its own micro-steps in, in order, and sends it on; the last rank sends the total to every other rank once it has all of it. Large sums travel in chunks so the ranks work at the same time; chunking changes the pipelining, never the order of the additions. A rank after the first therefore keeps its micro-steps apart until the running sum reaches it (`ProcessGroup::grad_sum` gives that kind of sum; pre-adding them is refused).
4. **Identical optimizer steps.** Every rank applies the same optimizer to the same summed gradient, so the vars stay identical without being sent. `check_in_sync` verifies it, and every coordinated checkpoint verifies it again.
5. **No timing in the result.** No compression, no quantised or asynchronous reductions, no arrival-order sums. Every message carries its collective's operation and sequence number; ranks that call different collectives fail at once.
6. **The key and pooling.** `ProcessGroup::key()` is the reproducibility key, `mode=deterministic order=flat-rank backend=tcp`. The world size is not part of it: by rules 1–3 the weights depend on the number of global micro-steps, not on how they are split among ranks (the gate below shows it), so runs at different world sizes may be pooled on the same platform key. The world size stays in every run's record (`ProcessGroup::record()`, and the checkpoint's meta). Before pooling on a new (platform, device) pair, a short run with per-step weight-hash checks at world sizes 1, 2 and 4 must show the identity there; a GPU needs its own showing. A resume still keeps the checkpoint's world size and job key (it refuses others): pooling compares finished runs, it does not move a run between world sizes.
7. **Order variants.** The flat order is the default and the only one built. A per-rank-first order (each rank pre-adds its micro-steps, then the ranks' sums are added in rank order) would save the deferred micro-steps' memory but makes the bits depend on the world size; if it is ever added, it is an opt-in whose key includes the world size, and it is never pooled across world sizes.
8. **Randomness inside a micro-step** (dropout) must be keyed by the global micro-step, not by the rank (the gate's model uses dropout 0.1 this way).
9. **Data.** Item p of the global stream is `perm_e[p mod n]` with e = p div n, where `perm_e` is a Fisher–Yates permutation from `rng::derive(seed, "shard epoch e")` (u64 draws, platform-independent). Global micro-batch j of step s starts at item `(s·M + j)·b`. Rank r takes micro-batches `r·k .. (r+1)·k`, so the ranks' micro-batches in rank order are the single process's. The loader cursor is the step number. `Shard::hash` is a sha256 of the shard's key (seed, items, micro-batch, micro-steps, rank, world size) and every item it takes; the training ranks all-gather their shard hashes and check every rank's against their own computation.
10. **A fast mode** (backend-native order, overlap, NCCL's reductions) is not built. It will carry its own key and never be mixed with this mode.

## Rendezvous and failure

- Rank 0 listens on `MASTER_ADDR:MASTER_PORT`. Every other rank connects and sends its rank, world size, job key and the address of its data listener (on the interface it reached rank 0 through). Rank 0 checks that the ranks are distinct and agree on the world size and the job key, then sends the address table, or the reason for refusing the job, to every rank. Ranks connect to every lower rank, accept every higher one, and a barrier ends the rendezvous.
- A lost peer is an error on the next message (end of stream), not a hang; a silent peer fails after `GroupOptions::timeout` (600 s by default).
- The launcher stops the whole job when any rank fails or is killed, and names the rank.
- Plain TCP, no encryption or authentication: private networks only.

## Coordinated checkpoints

Every rank hashes its vars, Adam's moments and step count; the hashes must agree. Rank 0 writes `step-<step>/` (`vars.json`, `adam_m.json`, `adam_v.json` in the bit-exact var-map format, and `meta.json` with the step, the world size, both keys, every rank's cursor, the state hash and each file's sha256) through a temporary folder and a rename, then points `latest` at it by a rename. A barrier ends the checkpoint, so once any rank is past it the checkpoint is on disk whole. Loading checks every hash and refuses another world size or job key. Rank 0 writes on its own machine: across machines the folder must be shared or copied before a resume.

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

**Refusals and failures:** another job key or world size at the rendezvous (refused on every rank, with the reason), mismatched collectives, a rank that leaves (its peers fail at once), a later rank that pre-added its micro-steps, a failing rank under the launcher (the job stops; the others are killed), a changed byte in a checkpoint (sha256 mismatch).

**The example** (`examples/ddp_copy_task.rs`, 200 steps, 4 global micro-steps of 8 items): final weights sha256 `7d4c24b808b50f964754c8c2e6a674ba7b82e91af96023a9649ad0ea22be6012` alone, under the launcher with 2 processes and with 4, on every rank.

The whole gate (20 tests) runs in about 3 s: `cargo test --release --lib nn::dist`.

## Single-device behaviour is unchanged

- No existing file changes behaviour. The changes outside `nn::dist`: the module line in `nn/mod.rs`, a new error variant `NnError::Dist` (kind "dist"), two examples, this note and a README bullet.
- All 88 existing library tests and the README doctests pass.
- Six training configurations (decoder, MoE decoder, decoder with dropout; each with AdamW and SGD with momentum; 60 steps, 3 models on the seed axis) give byte-identical final vars and logits built against 0.2.0 and against this branch; the copy-task example's output is identical.
- `data_parallel_step` with the single-process group and one micro-step equals `nn::train_step` bit for bit (vars and Adam moments, `one_process_with_one_micro_step_equals_the_plain_trainer`).

## For the release notes

- adds the variant NnError::Dist

## Known limits and next steps

- **f32 gradients only.** f64, f16 and bf16 gradients are refused by `GradSum`. bf16 with f32 master weights comes with mixed precision.
- **CPU only, tested on one machine.** GPU gradients would be copied to the host for the sum (correct, not fast). Multi-GPU and multi-machine runs need the later gates (2 nodes × 1 GPU, 1 node × 2 GPUs and 1 GPU × 2 micro-steps identical).
- **The OS maths library.** softmax, log_softmax and `powf` call the platform's `exp`, `log` and `pow`, whose last bits may differ between operating systems or library versions. The gate runs on one machine, so it is unaffected. Across machines, every node must use the same maths library (or the engine its own implementations); a difference would make the ranks' optimizer steps differ, which `check_in_sync` reports as an error rather than letting the ranks drift.
- **Throughput.** The running sum passes the ranks one after another (latency rises with W) and the last rank sends the total to each rank (its traffic rises with W). Both keep the order; a tree or ring broadcast of the total, bucketing and overlap with the backward pass are later work and must not change the order of the additions.
- **Memory.** A rank after the first keeps its k micro-step gradients until the running sum arrives (k copies of the gradients; with one micro-step per rank, none extra).
- **Checkpoints.** Rank 0 writes; multi-machine resume needs a shared or copied folder. The optimizer state saved is Adam's; SGD's momentum buffers are not yet in the checkpoint.
- **Not yet built:** reduce-scatter, sharded optimizer state and parameters (FSDP-style), elastic restart, a fast mode, generic callbacks (batch seen, checkpoint reached), NCCL.
