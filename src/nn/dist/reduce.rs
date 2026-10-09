//! The one ordered gradient sum, shared by local accumulation and the all-reduce.
//!
//! A step's gradient is the left fold, over its micro-steps in order, of their gradient sets:
//! `((g₀ + g₁) + g₂) + …`, elementwise in f32, per var. The fold starts from the first micro-step
//! that has a gradient for the var (never from zeros, which would turn a −0 into +0), and a
//! micro-step without one (the var took no part) adds nothing; a var no micro-step touched stays
//! `None`. `Reduce::Mean` then divides by the number of micro-steps, as `Tensor::div_scalar` does
//! (skipped when there was one, so a single micro-step passes its gradient through bit for bit).
//!
//! Across ranks the micro-steps are ordered by rank, then by their order on the rank: rank r's
//! i-th of k micro-steps is the global micro-step `r·k + i`. One process with W·k micro-steps and
//! W processes with k each therefore fold the same values in the same order.

use crate::error::{NnError, Result};
use crate::nn::VarMap;
use crate::tensor::{DType, Device, Tensor};

/// One var's place in a gradient set: its shape and the device its gradient goes back to.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Slot {
    pub shape: Vec<usize>,
    pub device: Device,
}

impl Slot {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A gradient set on the host: per var, its f32 values or None.
pub(crate) type Part = Vec<Option<Vec<f32>>>;

/// Fold `part` into `acc`, var by var: the one addition of the ordered sum.
pub(crate) fn fold(acc: &mut [Option<Vec<f32>>], part: &[Option<Vec<f32>>]) {
    debug_assert_eq!(acc.len(), part.len());
    for (a, p) in acc.iter_mut().zip(part) {
        fold_var(a, p.as_deref());
    }
}

/// Fold one var's contribution into its running sum.
pub(crate) fn fold_var(acc: &mut Option<Vec<f32>>, part: Option<&[f32]>) {
    match (acc.as_mut(), part) {
        (Some(a), Some(p)) => {
            for (x, y) in a.iter_mut().zip(p) {
                *x += *y;
            }
        }
        (None, Some(p)) => *acc = Some(p.to_vec()),
        _ => {}
    }
}

/// How a step's summed gradient is scaled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Reduce {
    /// The sum over the micro-steps.
    Sum,
    /// The sum divided by the number of micro-steps (torch DDP's default, for mean losses).
    #[default]
    Mean,
}

/// The ordered sum of a step's micro-step gradients, for the vars of one `VarMap`.
///
/// `GradSum::new` folds each micro-step in as it is added (one process, or the first rank).
/// A rank after the first must keep its micro-steps apart until the running sum of the ranks
/// before it arrives (`ProcessGroup::grad_sum` gives the right kind for the rank): the fold
/// order is global, so it cannot pre-add its own micro-steps. With one micro-step per rank the
/// two kinds hold the same.
#[derive(Clone, Debug)]
pub struct GradSum {
    slots: Vec<Slot>,
    /// sha256 of the vars' names and shapes: ranks check they sum the same layout.
    layout: String,
    /// The folded sum (`deferred` false).
    acc: Option<Part>,
    /// The micro-steps not yet folded, in order (`deferred` true).
    pending: Vec<Part>,
    count: usize,
    deferred: bool,
}

impl GradSum {
    /// A sum over the vars of `vars` that folds each micro-step in as it is added.
    pub fn new(vars: &VarMap) -> GradSum {
        let slots: Vec<Slot> = vars.tensors().iter().map(|t| Slot { shape: t.shape().to_vec(), device: t.device() }).collect();
        let mut text = String::new();
        for (n, s) in vars.names().iter().zip(&slots) {
            text.push_str(&format!("{n}:{:?};", s.shape));
        }
        GradSum { slots, layout: crate::hash::sha256_hex(text.as_bytes()), acc: None, pending: Vec::new(), count: 0, deferred: false }
    }

    /// A sum that keeps its micro-steps apart until they are folded into a running sum.
    pub(crate) fn deferred(vars: &VarMap) -> GradSum {
        GradSum { deferred: true, ..GradSum::new(vars) }
    }

    /// The micro-steps added (or, after an all-reduce, folded over every rank).
    pub fn count(&self) -> usize {
        self.count
    }

    /// Whether the micro-steps are kept apart (a rank after the first).
    pub fn is_deferred(&self) -> bool {
        self.deferred
    }

    /// Add one micro-step's gradients, by var index as `VarMap::grads` gives them (None: the var
    /// took no part). Each gradient must be f32 and have its var's shape.
    pub fn add(&mut self, grads: Vec<Option<Tensor>>) -> Result<()> {
        if grads.len() != self.slots.len() {
            return Err(NnError::Dist(format!("{} gradient slots for {} vars", grads.len(), self.slots.len())));
        }
        let mut part: Part = Vec::with_capacity(grads.len());
        for (i, (g, s)) in grads.iter().zip(&self.slots).enumerate() {
            match g {
                Some(t) => {
                    if t.dtype() != DType::F32 {
                        return Err(NnError::Dist(format!("var {i}: a {} gradient (the ordered sum takes f32 gradients)", t.dtype().name())));
                    }
                    if t.shape() != s.shape.as_slice() {
                        return Err(NnError::Dist(format!("var {i}: a gradient of shape {:?} for a var of shape {:?}", t.shape(), s.shape)));
                    }
                    part.push(Some(t.to_vec()));
                }
                None => part.push(None),
            }
        }
        self.push(part);
        Ok(())
    }

    fn push(&mut self, part: Part) {
        self.count += 1;
        if self.deferred {
            self.pending.push(part);
        } else {
            match &mut self.acc {
                None => self.acc = Some(part),
                Some(acc) => fold(acc, &part),
            }
        }
    }

    pub(crate) fn slots(&self) -> &[Slot] {
        &self.slots
    }

    pub(crate) fn layout(&self) -> &str {
        &self.layout
    }

    /// The local contributions, in order, for the all-reduce: the folded sum (one entry) or the
    /// kept micro-steps; and the count.
    pub(crate) fn take_parts(&mut self) -> (Vec<Part>, usize) {
        let parts = if self.deferred { std::mem::take(&mut self.pending) } else { self.acc.take().into_iter().collect() };
        (parts, std::mem::replace(&mut self.count, 0))
    }

    /// This sum, replaced by a folded sum of `count` micro-steps (the all-reduce's result).
    pub(crate) fn set_folded(&mut self, acc: Part, count: usize) {
        self.acc = Some(acc);
        self.pending.clear();
        self.count = count;
        self.deferred = false;
    }

    /// The step's gradients, by var index, on each var's device: the folded sum scaled by `how`
    /// (a deferred sum is folded first, as one process would).
    pub fn finish(mut self, how: Reduce) -> Vec<Option<Tensor>> {
        let count = self.count;
        if self.deferred {
            let mut acc: Option<Part> = None;
            for p in std::mem::take(&mut self.pending) {
                match &mut acc {
                    None => acc = Some(p),
                    Some(a) => fold(a, &p),
                }
            }
            self.acc = acc;
        }
        let Some(acc) = self.acc else { return vec![None; self.slots.len()] };
        self.slots
            .into_iter()
            .zip(acc)
            .map(|(s, v)| {
                v.map(|mut v| {
                    if how == Reduce::Mean && count != 1 {
                        let c = count as f32;
                        for x in v.iter_mut() {
                            *x /= c;
                        }
                    }
                    Tensor::from_data(v, s.shape).to(s.device)
                })
            })
            .collect()
    }
}
