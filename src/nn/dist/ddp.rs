//! Data-parallel training steps: every rank runs its micro-steps, the gradients are summed in
//! global order, and every rank takes the same optimizer step on the same sum, so the vars stay
//! identical on every rank without being sent.

use super::group::ProcessGroup;
use super::reduce::Reduce;
use crate::error::{NnError, Result};
use crate::nn::{Optimizer, VarMap};
use crate::tensor::Tensor;

/// sha256 of the vars' saved form (names, dtypes, shapes and every bit of their values).
pub fn state_hash(vars: &VarMap) -> Result<String> {
    Ok(crate::hash::sha256_hex(vars.to_json()?.as_bytes()))
}

/// Check that every rank holds the same vars, bit for bit; returns their hash. A rank whose vars
/// differ fails the check on every rank. The group counts the checks for its record.
pub fn check_in_sync(group: &mut ProcessGroup, vars: &VarMap) -> Result<String> {
    let h = state_hash(vars)?;
    let all = group.all_gather(h.as_bytes());
    let all = match all {
        Ok(a) => a,
        Err(e) => {
            group.checks.failed = true;
            return Err(e);
        }
    };
    if let Some(q) = all.iter().position(|x| x.as_slice() != h.as_bytes()) {
        group.checks.failed = true;
        let theirs = String::from_utf8_lossy(&all[q]).to_string();
        return Err(NnError::Dist(format!("rank {}: the vars differ from rank {q}'s ({h} against {theirs})", group.rank())));
    }
    group.checks.passed += 1;
    group.checks.last = Some(h.clone());
    Ok(h)
}

/// One data-parallel optimizer step: `data_parallel_grads`, then `opt` steps with the summed
/// gradients on every rank. Returns the number of micro-steps summed over every rank.
///
/// With the single-process group this is gradient accumulation over `micro_steps_per_rank`
/// micro-steps, through the same sum: W ranks with k micro-steps each and one process with W·k
/// give the same vars bit for bit when the ranks' micro-batches, in rank order, are the
/// process's (`Shard` gives them so). Anything random inside a micro-step (dropout) must be
/// drawn from the global micro-step's index, not the rank's.
pub fn data_parallel_step(
    group: &mut ProcessGroup,
    vars: &mut VarMap,
    opt: &mut dyn Optimizer,
    micro_steps_per_rank: usize,
    how: Reduce,
    lr_scale: f64,
    micro: impl FnMut(&VarMap, usize) -> Tensor,
) -> Result<usize> {
    let (grads, count) = data_parallel_grads(group, vars, micro_steps_per_rank, how, micro)?;
    opt.step(vars, grads, lr_scale);
    Ok(count)
}

/// The summed gradients of one data-parallel step, the same on every rank, by var index (None:
/// no micro-step on any rank used the var), and the number of micro-steps summed; for a caller
/// that transforms them (clips them, say) before its optimizer steps, as every rank must then
/// do identically.
///
/// The vars must live on the group's device (`GroupOptions::device`). For each of this rank's
/// `micro_steps_per_rank` micro-steps i, `micro(lifted, i)` builds the loss to differentiate from
/// the lifted vars (a fresh `VarMap::lifted` per micro-step). Each micro-step's gradients are
/// copied to the host as f32 and summed there by `GradSum` in the flat rank order, whatever the
/// device; the sum is scaled by `how` and goes back to the vars' devices.
pub fn data_parallel_grads(
    group: &mut ProcessGroup,
    vars: &VarMap,
    micro_steps_per_rank: usize,
    how: Reduce,
    mut micro: impl FnMut(&VarMap, usize) -> Tensor,
) -> Result<(Vec<Option<Tensor>>, usize)> {
    // The run's key names the group's device: the vars must live there.
    if let Some((i, t)) = vars.tensors().iter().enumerate().find(|(_, t)| t.device() != group.device()) {
        return Err(NnError::Dist(format!("var {:?} is on {:?}, the group trains on {:?} (GroupOptions::device)", vars.names()[i], t.device(), group.device())));
    }
    let mut sum = group.grad_sum(vars);
    for i in 0..micro_steps_per_rank {
        let lifted = vars.lifted();
        let loss = micro(&lifted, i);
        let grads = loss.backward();
        sum.add(lifted.grads(&grads))?;
    }
    group.all_reduce_grads(&mut sum)?;
    let count = sum.count();
    if count == 0 {
        return Err(NnError::Dist("a step without micro-steps on any rank".into()));
    }
    Ok((sum.finish(how), count))
}
