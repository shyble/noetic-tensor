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

/// One data-parallel optimizer step.
///
/// For each of this rank's `micro_steps_per_rank` micro-steps i, `micro(lifted, i)` builds the loss
/// to differentiate from the lifted vars (a fresh `VarMap::lifted` per micro-step); its gradients
/// go into the rank's `GradSum`. The sums are all-reduced in global order and scaled by `how`,
/// and `opt` takes one step with them. Returns the number of micro-steps summed over every rank.
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
    mut micro: impl FnMut(&VarMap, usize) -> Tensor,
) -> Result<usize> {
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
    opt.step(vars, sum.finish(how), lr_scale);
    Ok(count)
}
