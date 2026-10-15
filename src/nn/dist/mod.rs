//! Data-parallel training across processes (tested on one machine; multi-machine runs are
//! untested), in the shape of torch.distributed, DDP and torchrun: a process group initialised
//! from the environment, collectives over TCP (std only), a gradient sum shared by local
//! accumulation and the all-reduce, a sharded sampler, a coordinated checkpoint and a launcher.
//!
//! **Deterministic mode** (the only mode so far):
//! - every all-reduce adds the ranks' contributions in rank order, and each rank's micro-steps in
//!   their order: the sum over a step is the left fold over the global micro-steps
//!   `j = rank · k + i` (k micro-steps per rank), starting from the first contribution;
//! - so W ranks with k micro-steps each give the same bits as one process with W·k micro-steps
//!   (`GradSum` is the one routine both paths use, and `ProcessGroup::all_reduce_grads` passes the
//!   running sum along the ranks in order);
//! - no compression, no quantised or asynchronous reductions, no timing-dependent order;
//! - the world size is not part of the run's key (`ProcessGroup::key`): the weights depend on the
//!   number of global micro-steps, not on how the ranks share them; it stays in the run's record
//!   (`ProcessGroup::record`), and a checkpoint refuses to resume at another world size.
//!
//! **Devices.** Each rank trains on one device (`GroupOptions::device`, from `LOCAL_RANK` through
//! `DistEnv::device`): the CPU reference backend, Metal or CUDA. Gradients are summed on the host
//! in the same order whatever the device; a GPU's kind, model and versions enter the key
//! (`device_key`), so CPU and GPU runs never share one. Tested on one machine per GPU kind, with
//! several processes on one GPU; multi-GPU and multi-machine runs are untested.
//!
//! **Fast mode** (`DistMode::Fast`, opt-in through `GroupOptions::mode`): per-rank sums, gradient
//! buckets, a ring all-reduce (a ring reduce-scatter for sharded training) and communication
//! overlapped with the backward pass of each rank's last micro-step. Its key says `mode=fast`
//! with the world size and the buckets, its record says "fast mode, not reproducible bit for
//! bit", and it is never pooled with or compared byte for byte against this mode: a job mixing
//! modes is refused at the rendezvous, and `ProcessGroup::require_deterministic` refuses it
//! wherever a caller needs the deterministic mode.
//!
//! The environment is torchrun's: `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`,
//! `LOCAL_RANK`, `LOCAL_WORLD_SIZE`, `GROUP_RANK` (the node). Without `RANK` and `WORLD_SIZE`
//! distribution is off: `ProcessGroup::from_env` gives the single-process group, which opens no
//! socket and whose collectives are the identity.

mod auth;
mod checkpoint;
mod ddp;
mod env;
mod fast;
mod group;
mod launch;
mod reduce;
mod shard;
mod wire;
mod zero;

#[cfg(test)]
mod tests;

pub use auth::JobSecret;
pub use checkpoint::{Checkpoint, CheckpointMeta, CheckpointOptimizer, CHECKPOINT_FORMAT};
pub use ddp::{check_in_sync, data_parallel_grads, data_parallel_step, state_hash};
pub use env::{device_key, platform_key, DeviceKind, DistEnv};
pub use fast::{DistMode, FastOptions};
pub use group::{wire_version, GroupOptions, ProcessGroup, WIRE_PROTOCOL};
pub use launch::{free_port, run, spawn, Job, LaunchConfig};
pub use reduce::{GradSum, Reduce};
pub use shard::{Shard, ShardSpec};
pub use zero::{Partition, Piece, ShardCheckpoint, ShardCheckpointMeta, ShardGrads, ShardLayout, ShardMemory, ShardOptions, ShardRank, ShardStage, ShardedDataParallel, SHARD_CHECKPOINT_FORMAT};
