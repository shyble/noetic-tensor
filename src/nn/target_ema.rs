//! The target network of a self-distillation or joint-embedding objective: a moving average
//! of the online vars that is initialised as a copy of them, updated
//! `θ′ ← τ·θ′ + (1 − τ)·θ` with a scheduled τ, without bias correction, and read detached (the
//! target never receives gradients).
//!
//! It is separate from `nn::ema::Ema`, which starts at zero and divides by 1 − decayᵗ (an average
//! of the weights for evaluation); here θ′ is the copy itself from step 0.
//!
//! The schedule is BYOL's cosine: `τ_k = end − (end − start)·(cos(π·k/K) + 1)/2` for the update
//! at step k < K, and `end` from step K on (`TauSchedule::cosine(K)`: 0.996 → 1). joint-embedding work's own
//! schedule is linear from 0.996 to 1 (not built here). There is no
//! default schedule: a cosine needs its step count K ≥ 1 from the caller (a default K would freeze
//! the target after K updates), and `TargetEma::new` refuses K = 0.
//!
//! ```compile_fail
//! let _ = noetic::nn::TauSchedule::default(); // no Default: the step count is required
//! ```
//!
//! Each update is two scalar
//! multiplications and an addition per var, on the vars' device and in their dtype, so it is
//! deterministic wherever those operations are.

use super::var::{VarMap, VarMapFile};
use crate::error::{NnError, Result};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// The τ schedule of a `TargetEma`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TauSchedule {
    /// τ fixed.
    Constant(f64),
    /// From `start` at step 0 to `end` at step `steps` (and after) along a half cosine.
    Cosine { start: f64, end: f64, steps: u64 },
}

impl TauSchedule {
    /// The cosine schedule 0.996 → 1 over `steps` ≥ 1 updates (the length of training).
    pub fn cosine(steps: u64) -> Self {
        assert!(steps > 0, "a cosine τ schedule needs its step count (≥ 1)");
        TauSchedule::Cosine { start: 0.996, end: 1.0, steps }
    }

    /// τ for the update at step `k` (counted from 0).
    pub fn tau(&self, k: u64) -> f64 {
        match *self {
            TauSchedule::Constant(t) => t,
            TauSchedule::Cosine { start, end, steps } => {
                // The endpoints exactly.
                if k >= steps {
                    return end;
                }
                if k == 0 {
                    return start;
                }
                let frac = k as f64 / steps as f64;
                end - (end - start) * ((std::f64::consts::PI * frac).cos() + 1.0) / 2.0
            }
        }
    }
}

/// The saved form of a `TargetEma` (`FORMAT`): the step count, the schedule (f64 values as their
/// bit patterns, so a restore is exact) and θ′.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetEmaFile {
    pub format: String,
    pub step: u64,
    /// "constant" or "cosine".
    pub schedule: String,
    /// `[τ]` or `[start, end]`, as `f64::to_bits`.
    pub tau_bits: Vec<u64>,
    /// The cosine's step count (0 for a constant).
    pub steps: u64,
    pub target: VarMapFile,
}

const FORMAT: &str = "nn.target_ema.v1";

pub struct TargetEma {
    pub schedule: TauSchedule,
    target: VarMap,
    step: u64,
}

impl TargetEma {
    /// θ′ = a copy of `online` (untracked), at step 0.
    /// Panics on a cosine schedule of 0 steps or a τ outside [0, 1].
    pub fn new(online: &VarMap, schedule: TauSchedule) -> Self {
        match schedule {
            TauSchedule::Constant(t) => assert!((0.0..=1.0).contains(&t), "τ = {t} outside [0, 1]"),
            TauSchedule::Cosine { start, end, steps } => {
                assert!(steps > 0, "a cosine τ schedule needs its step count (≥ 1)");
                assert!((0.0..=1.0).contains(&start) && (0.0..=1.0).contains(&end), "τ {start} → {end} outside [0, 1]");
            }
        }
        TargetEma { schedule, target: online.detached(), step: 0 }
    }

    /// Updates done so far (the step of the next update).
    pub fn step(&self) -> u64 {
        self.step
    }

    /// τ of the next update.
    pub fn tau(&self) -> f64 {
        self.schedule.tau(self.step)
    }

    /// θ′ ← τ·θ′ + (1 − τ)·θ at the scheduled τ, then the step advances. Returns the τ used.
    pub fn update(&mut self, online: &VarMap) -> f64 {
        let tau = self.tau();
        self.update_with(online, tau);
        tau
    }

    /// θ′ ← τ·θ′ + (1 − τ)·θ at the given τ ∈ [0, 1]; the step advances. τ = 1 keeps θ′ and
    /// τ = 0 copies θ, exactly (for finite values).
    pub fn update_with(&mut self, online: &VarMap, tau: f64) {
        assert!((0.0..=1.0).contains(&tau), "τ = {tau} outside [0, 1]");
        assert_eq!(online.names(), self.target.names(), "the target is for another var map");
        let mut next = VarMap::new();
        for ((name, t), x) in self.target.iter().zip(online.tensors()) {
            assert_eq!(t.shape(), x.shape(), "var {name:?}: target shape {:?} vs online {:?}", t.shape(), x.shape());
            let x = x.clone().untracked();
            let v = t.clone().mul_scalar(tau) + x.mul_scalar(1.0 - tau);
            next.insert(name, v.untracked()).expect("distinct names");
        }
        self.target = next;
        self.step += 1;
    }

    /// θ′, detached: untracked tensors, so a forward through them records nothing for the
    /// backward pass and the target never receives gradients.
    pub fn target(&self) -> VarMap {
        self.target.detached()
    }

    /// The saved form.
    pub fn to_file(&self) -> Result<TargetEmaFile> {
        let (schedule, tau_bits, steps) = match self.schedule {
            TauSchedule::Constant(t) => ("constant", vec![t.to_bits()], 0),
            TauSchedule::Cosine { start, end, steps } => ("cosine", vec![start.to_bits(), end.to_bits()], steps),
        };
        Ok(TargetEmaFile { format: FORMAT.to_string(), step: self.step, schedule: schedule.to_string(), tau_bits, steps, target: self.target.to_file()? })
    }

    /// Rebuild from the saved form (θ′ on the calling thread's default device).
    pub fn from_file(f: &TargetEmaFile) -> Result<TargetEma> {
        if f.format != FORMAT {
            return Err(NnError::Persist(format!("unknown target EMA format {:?} (expected {FORMAT:?})", f.format)));
        }
        let b = |i: usize| f.tau_bits.get(i).map(|x| f64::from_bits(*x)).ok_or_else(|| NnError::Persist("target EMA: missing τ".to_string()));
        let schedule = match f.schedule.as_str() {
            "constant" => TauSchedule::Constant(b(0)?),
            "cosine" => TauSchedule::Cosine { start: b(0)?, end: b(1)?, steps: f.steps },
            other => return Err(NnError::Persist(format!("target EMA: unknown schedule {other:?}"))),
        };
        Ok(TargetEma { schedule, target: VarMap::from_file(&f.target)?, step: f.step })
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(&self.to_file()?).map_err(|e| NnError::Persist(format!("target EMA: {e}")))
    }

    pub fn from_json(s: &str) -> Result<TargetEma> {
        let f: TargetEmaFile = serde_json::from_str(s).map_err(|e| NnError::Persist(format!("target EMA: {e}")))?;
        TargetEma::from_file(&f)
    }
}

/// `x` untracked (a detached read of a tensor computed from the target).
pub fn stop_gradient(x: &Tensor) -> Tensor {
    x.clone().untracked()
}
