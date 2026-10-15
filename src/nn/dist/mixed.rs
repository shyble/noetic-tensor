//! Mixed-precision steps: the matrix products at `MatmulPrecision::Half` (16-bit operands on the
//! GPU's matrix units, f32 accumulation; `tensor::set_matmul_precision`), with f32 vars (the master
//! weights), f32 gradients, f32 optimizer state and a static loss scale (`LossScale`), for data
//! parallelism and sharded data parallelism, in either distributed mode.
//!
//! - **Keys.** The group must be made at `Half` (its key then ends with the product's part, so a
//!   mixed run never shares a key with an f32 run: checkpoints and rendezvous refuse the other);
//!   a mixed step on a group made at f32 is refused. In the fast mode the key is the fast one with
//!   the same part.
//! - **A step.** Each micro-step's loss is scaled by 2^k before its backward pass; the summed
//!   gradients (or this rank's pieces) are scaled back by 2^−k, which is exact for a power of two;
//!   if any gradient element on any rank is not finite, every rank skips the optimizer step (the
//!   ranks agree by an all-gather), else every rank takes it as usual. Nothing else changes: the
//!   sum's order is the mode's (deterministic or fast), the optimizer's arithmetic is f32.

use super::ddp::data_parallel_grads;
use super::group::ProcessGroup;
use super::reduce::Reduce;
use super::zero::ShardedDataParallel;
use crate::error::{NnError, Result};
use crate::nn::loss_scale::{all_finite, LossScale};
use crate::nn::{Optimizer, VarMap};
use crate::tensor::{MatmulPrecision, Tensor};

/// What a mixed-precision step did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MixedStep {
    /// The micro-steps summed over every rank.
    pub count: usize,
    /// Whether the optimizer stepped (false: a gradient was not finite on some rank; skipped).
    pub stepped: bool,
}

fn require_half(group: &ProcessGroup) -> Result<()> {
    if group.matmul_precision() != MatmulPrecision::Half {
        return Err(NnError::Dist(format!(
            "a mixed-precision step on a group made at matmul precision {} (set MatmulPrecision::Half with tensor::set_matmul_precision before the group is made, so that its key names it)",
            group.matmul_precision().name()
        )));
    }
    Ok(())
}

/// Every rank's `finite`, agreed: true only when it is true on every rank.
fn agree_finite(group: &mut ProcessGroup, finite: bool) -> Result<bool> {
    Ok(group.all_gather(&[finite as u8])?.iter().all(|f| f.as_slice() == [1]))
}

/// The summed gradients of a mixed-precision data-parallel step (as `data_parallel_grads`, the
/// losses scaled by `scale` and the sums scaled back), and whether every element on every rank is
/// finite. For a caller that clips them before its optimizer steps (when they are finite).
pub fn mixed_data_parallel_grads(group: &mut ProcessGroup, vars: &VarMap, micro_steps_per_rank: usize, how: Reduce, scale: LossScale, mut micro: impl FnMut(&VarMap, usize) -> Tensor) -> Result<(Vec<Option<Tensor>>, usize, bool)> {
    require_half(group)?;
    let (mut grads, count) = data_parallel_grads(group, vars, micro_steps_per_rank, how, |l, i| scale.scale(micro(l, i)))?;
    scale.unscale(&mut grads);
    let finite = agree_finite(group, all_finite(&grads))?;
    Ok((grads, count, finite))
}

/// One mixed-precision data-parallel step: `mixed_data_parallel_grads`, then the optimizer's step
/// on every rank when every gradient is finite (else every rank skips it).
#[allow(clippy::too_many_arguments)]
pub fn mixed_data_parallel_step(group: &mut ProcessGroup, vars: &mut VarMap, opt: &mut dyn Optimizer, micro_steps_per_rank: usize, how: Reduce, lr_scale: f64, scale: LossScale, micro: impl FnMut(&VarMap, usize) -> Tensor) -> Result<MixedStep> {
    let (grads, count, finite) = mixed_data_parallel_grads(group, vars, micro_steps_per_rank, how, scale, micro)?;
    if finite {
        opt.step(vars, grads, lr_scale);
    }
    Ok(MixedStep { count, stepped: finite })
}

/// One mixed-precision sharded step: `ShardedDataParallel::grads` with the losses scaled, this
/// rank's pieces scaled back, the ranks' finiteness agreed, then `apply` on every rank (or a skip
/// on every rank).
#[allow(clippy::too_many_arguments)]
pub fn mixed_sharded_step<O: Optimizer>(sdp: &mut ShardedDataParallel<O>, group: &mut ProcessGroup, micro_steps_per_rank: usize, how: Reduce, lr_scale: f64, scale: LossScale, mut micro: impl FnMut(&VarMap, usize) -> Tensor) -> Result<MixedStep> {
    require_half(group)?;
    let mut grads = sdp.grads(group, micro_steps_per_rank, how, |l, i| scale.scale(micro(l, i)))?;
    scale.unscale(grads.grads_mut());
    let finite = agree_finite(group, all_finite(grads.grads()))?;
    let count = grads.count();
    if finite {
        sdp.apply(group, grads, lr_scale)?;
    }
    Ok(MixedStep { count, stepped: finite })
}

/// A mixed run's record: the group's record (its key names the product), the precision and the
/// loss scale.
pub fn mixed_record(group: &ProcessGroup, scale: LossScale) -> String {
    format!("{} matmul={} {}", group.record(), group.matmul_precision().name(), scale.name())
}
